import copy
import math
import os.path as osp
import random
import re
from collections import defaultdict

import numpy as np
import torch.distributed as dist
from torch.utils.data.sampler import Sampler


def extract_floor_from_path(img_path):
    """
    이미지 경로에서 층 정보 추출

    Args:
        img_path: 이미지 경로

    Returns:
        str: 층 정보 (예: "1F", "2F", "3F", "4F") 또는 None
    """
    filename = osp.basename(img_path)
    # 파일명에서 날짜 다음의 층 정보 추출
    # 패턴: 날짜_층정보_...
    match = re.match(r"\d{4}-\d{2}-\d{2}_([^_]+)_", filename)
    if match:
        floor = match.group(1)
        # 층 정보가 "1F", "2F", "3F", "4F" 형식인지 확인
        if re.match(r"\d+F(-.*)?", floor):
            return floor
    return None


_LOCAL_PROCESS_GROUP = None
import pickle

import torch


def _get_global_gloo_group():
    """
    Return a process group based on gloo backend, containing all the ranks
    The result is cached.
    """
    if dist.get_backend() == "nccl":
        return dist.new_group(backend="gloo")
    else:
        return dist.group.WORLD


def _serialize_to_tensor(data, group):
    backend = dist.get_backend(group)
    assert backend in ["gloo", "nccl"]
    device = torch.device("cpu" if backend == "gloo" else "cuda")

    buffer = pickle.dumps(data)
    if len(buffer) > 1024**3:
        print(
            "Rank {} trying to all-gather {:.2f} GB of data on device {}".format(
                dist.get_rank(), len(buffer) / (1024**3), device
            )
        )
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to(device=device)
    return tensor


def _pad_to_largest_tensor(tensor, group):
    """
    Returns:
        list[int]: size of the tensor, on each rank
        Tensor: padded tensor that has the max size
    """
    world_size = dist.get_world_size(group=group)
    assert (
        world_size >= 1
    ), "comm.gather/all_gather must be called from ranks within the given group!"
    local_size = torch.tensor([tensor.numel()], dtype=torch.int64, device=tensor.device)
    size_list = [
        torch.zeros([1], dtype=torch.int64, device=tensor.device)
        for _ in range(world_size)
    ]
    dist.all_gather(size_list, local_size, group=group)
    size_list = [int(size.item()) for size in size_list]

    max_size = max(size_list)

    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    if local_size != max_size:
        padding = torch.zeros(
            (max_size - local_size,), dtype=torch.uint8, device=tensor.device
        )
        tensor = torch.cat((tensor, padding), dim=0)
    return size_list, tensor


def all_gather(data, group=None):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors).
    Args:
        data: any picklable object
        group: a torch process group. By default, will use a group which
            contains all ranks on gloo backend.
    Returns:
        list[data]: list of data gathered from each rank
    """
    if dist.get_world_size() == 1:
        return [data]
    if group is None:
        group = _get_global_gloo_group()
    if dist.get_world_size(group) == 1:
        return [data]

    tensor = _serialize_to_tensor(data, group)

    size_list, tensor = _pad_to_largest_tensor(tensor, group)
    max_size = max(size_list)

    # receiving Tensor from all ranks
    tensor_list = [
        torch.empty((max_size,), dtype=torch.uint8, device=tensor.device)
        for _ in size_list
    ]
    dist.all_gather(tensor_list, tensor, group=group)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def shared_random_seed():
    """
    Returns:
        int: a random number that is the same across all workers.
            If workers need a shared RNG, they can use this shared seed to
            create one.
    All workers must call this function, otherwise it will deadlock.
    """
    # DDP가 초기화되지 않았거나 단일 프로세스인 경우
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return np.random.randint(2**31)

    try:
        ints = np.random.randint(2**31)
        all_ints = all_gather(ints)
        return all_ints[0]
    except Exception as e:
        # all_gather 실패 시 rank 0의 seed를 사용하도록 fallback
        # 하지만 이 경우 동기화가 보장되지 않으므로 경고 출력
        if dist.get_rank() == 0:
            print(f"Warning: shared_random_seed() failed with {e}, using rank 0 seed")
        seed = np.random.randint(2**31)
        # rank 0의 seed를 broadcast
        if dist.get_rank() == 0:
            seed_tensor = torch.tensor([seed], dtype=torch.long, device="cpu")
        else:
            seed_tensor = torch.tensor([0], dtype=torch.long, device="cpu")
        dist.broadcast(seed_tensor, src=0)
        return int(seed_tensor.item())


class RandomIdentitySampler_DDP(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size
        self.world_size = dist.get_world_size()
        self.num_instances = num_instances
        self.mini_batch_size = self.batch_size // self.world_size
        self.num_pids_per_batch = self.mini_batch_size // self.num_instances

        # num_pids_per_batch가 0이면 배치 생성이 불가능하므로 최소 1로 보장
        if self.num_pids_per_batch == 0:
            # mini_batch_size가 num_instances보다 작은 경우
            # 최소한 1개의 identity를 사용하도록 조정
            self.num_pids_per_batch = 1
            # mini_batch_size도 조정 (num_pids_per_batch * num_instances)
            self.mini_batch_size = self.num_pids_per_batch * self.num_instances
            if dist.get_rank() == 0:
                print(
                    f"WARNING: num_pids_per_batch was 0. "
                    f"Adjusted to {self.num_pids_per_batch} "
                    f"(mini_batch_size adjusted to {self.mini_batch_size})"
                )

        # 디버깅: batch size 설정 확인
        if dist.get_rank() == 0:
            print(f"RandomIdentitySampler_DDP initialization:")
            print(f"  - batch_size (ims_per_batch): {self.batch_size}")
            print(f"  - world_size: {self.world_size}")
            print(f"  - mini_batch_size per GPU: {self.mini_batch_size}")
            print(f"  - num_instances: {self.num_instances}")
            print(f"  - num_pids_per_batch: {self.num_pids_per_batch}")
            print(
                f"  - Effective batch size per GPU: {self.num_pids_per_batch * self.num_instances}"
            )

        self.index_dic = defaultdict(list)

        for index, (_, pid, _, _) in enumerate(self.data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

        self.rank = dist.get_rank()
        # self.world_size = dist.get_world_size()
        self.length //= self.world_size

        # __len__에서 계산한 길이를 캐싱하기 위한 변수
        self._cached_length = None

    def __iter__(self):
        """
        Lightning 표준: Sampler는 인덱스를 하나씩 반환해야 함.
        BatchSampler가 이를 mini_batch_size 개씩 묶어서 배치로 만듦.
        """
        seed = shared_random_seed()
        np.random.seed(seed)
        self._seed = int(seed)
        final_idxs = self.sample_list()
        length = int(math.ceil(len(final_idxs) * 1.0 / self.world_size))
        final_idxs = self.__fetch_current_node_idxs(final_idxs, length)
        self.length = len(final_idxs)
        # __len__에서 사용할 수 있도록 캐시 업데이트
        self._cached_length = self.length

        # 디버깅: 인덱스 리스트 확인 (첫 번째 epoch만)
        if dist.get_rank() == 0 and not hasattr(self, "_debug_printed"):
            self._debug_printed = True
            print(
                f"DEBUG: RandomIdentitySampler_DDP.__iter__ - final_idxs length: {len(final_idxs)}"
            )
            print(
                f"DEBUG: RandomIdentitySampler_DDP.__iter__ - first 10 indices: {final_idxs[:10] if len(final_idxs) >= 10 else final_idxs}"
            )
            print(
                f"DEBUG: RandomIdentitySampler_DDP.__iter__ - mini_batch_size: {self.mini_batch_size}"
            )

        # Lightning 표준: 인덱스를 하나씩 반환 (BatchSampler가 묶음)
        for idx in final_idxs:
            yield idx

    def __fetch_current_node_idxs(self, final_idxs, length):
        """
        각 rank에 대해 연속된 인덱스 블록을 할당.
        BatchSampler가 제대로 작동하도록 인덱스 리스트를 반환.
        """
        total_num = len(final_idxs)
        # 각 rank가 받을 인덱스 범위 계산
        start_idx = self.rank * length
        end_idx = min(start_idx + length, total_num)

        # 연속된 인덱스 블록 반환 (BatchSampler가 이를 mini_batch_size 개씩 묶음)
        rank_idxs = final_idxs[start_idx:end_idx]
        return rank_idxs

    def sample_list(self):
        # np.random.seed(self._seed)
        avai_pids = copy.deepcopy(self.pids)
        batch_idxs_dict = {}

        batch_indices = []
        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = np.random.choice(
                avai_pids, self.num_pids_per_batch, replace=False
            ).tolist()
            for pid in selected_pids:
                if pid not in batch_idxs_dict:
                    idxs = copy.deepcopy(self.index_dic[pid])
                    if len(idxs) < self.num_instances:
                        idxs = np.random.choice(
                            idxs, size=self.num_instances, replace=True
                        ).tolist()
                    np.random.shuffle(idxs)
                    batch_idxs_dict[pid] = idxs

                avai_idxs = batch_idxs_dict[pid]
                # 남은 인덱스가 부족하면 복원 샘플링으로 채움
                if len(avai_idxs) < self.num_instances:
                    while len(avai_idxs) < self.num_instances:
                        avai_idxs.append(np.random.choice(self.index_dic[pid]))

                for _ in range(self.num_instances):
                    batch_indices.append(avai_idxs.pop(0))

                if len(avai_idxs) < self.num_instances:
                    avai_pids.remove(pid)

        # 남은 identity 처리: 가능한 만큼 배치 구성
        while len(avai_pids) > 0:
            num_available = min(len(avai_pids), self.num_pids_per_batch)
            if num_available == 0:
                break
            selected_pids = np.random.choice(
                avai_pids, num_available, replace=False
            ).tolist()
            temp_batch = []
            for pid in selected_pids:
                if pid not in batch_idxs_dict:
                    idxs = copy.deepcopy(self.index_dic[pid])
                    if len(idxs) < self.num_instances:
                        idxs = np.random.choice(
                            idxs, size=self.num_instances, replace=True
                        ).tolist()
                    np.random.shuffle(idxs)
                    batch_idxs_dict[pid] = idxs

                avai_idxs = batch_idxs_dict[pid]
                if len(avai_idxs) < self.num_instances:
                    while len(avai_idxs) < self.num_instances:
                        avai_idxs.append(np.random.choice(self.index_dic[pid]))

                for _ in range(self.num_instances):
                    temp_batch.append(avai_idxs.pop(0))

                if len(avai_idxs) < self.num_instances:
                    avai_pids.remove(pid)

            # 배치 크기가 맞으면 추가
            if len(temp_batch) == self.mini_batch_size:
                batch_indices.extend(temp_batch)

        return batch_indices

    def __len__(self):
        """
        Lightning 표준: Sampler는 인덱스 개수를 반환해야 함.
        BatchSampler가 배치 개수를 계산함.

        주의: __iter__에서 sample_list()를 호출하여 실제 인덱스 리스트를 생성하므로,
        __len__에서도 동일한 방식으로 계산해야 정확한 길이를 반환할 수 있습니다.
        하지만 sample_list()는 랜덤 샘플링을 수행하므로, 매번 다른 결과를 반환할 수 있습니다.

        해결책: sample_list()를 호출하여 실제 생성될 인덱스 개수를 계산합니다.
        계산 결과를 캐싱하여 오버헤드를 줄입니다.
        """
        # 캐시된 값이 있으면 사용
        if self._cached_length is not None:
            return self._cached_length

        # sample_list()를 호출하여 실제 생성될 인덱스 개수 계산
        # seed를 고정하여 일관된 결과 보장
        original_seed = random.getstate() if hasattr(random, "getstate") else None
        np_state = np.random.get_state()

        try:
            # 고정된 seed로 샘플링하여 일관된 길이 계산
            test_seed = 42  # 고정된 seed
            np.random.seed(test_seed)
            final_idxs = self.sample_list()
            length = int(math.ceil(len(final_idxs) * 1.0 / self.world_size))
            final_idxs = self.__fetch_current_node_idxs(final_idxs, length)
            self._cached_length = len(final_idxs)
            return self._cached_length
        finally:
            # random state 복원
            if original_seed is not None:
                random.setstate(original_seed)
            np.random.set_state(np_state)


class StratifiedIdentitySampler_DDP(Sampler):
    """
    층 정보를 고려한 PK 샘플러 (DDP 버전)

    각 identity에서 K개 샘플링 시, 층 정보가 있으면 각 층에서 균등하게 샘플링.
    층 정보가 없으면 기존 RandomIdentitySampler_DDP처럼 랜덤 샘플링 (fallback).

    Args:
        data_source (list): list of (img_path, pid, camid, trackid) or
                           (img_path, pid, camid, trackid, floor)
        batch_size (int): number of examples in a batch
        num_instances (int): number of instances per identity in a batch
    """

    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size
        self.world_size = dist.get_world_size()
        self.num_instances = num_instances
        self.mini_batch_size = self.batch_size // self.world_size
        self.num_pids_per_batch = self.mini_batch_size // self.num_instances

        # num_pids_per_batch가 0이면 배치 생성이 불가능하므로 최소 1로 보장
        if self.num_pids_per_batch == 0:
            # mini_batch_size가 num_instances보다 작은 경우
            # 최소한 1개의 identity를 사용하도록 조정
            self.num_pids_per_batch = 1
            # mini_batch_size도 조정 (num_pids_per_batch * num_instances)
            self.mini_batch_size = self.num_pids_per_batch * self.num_instances
            if dist.get_rank() == 0:
                print(
                    f"WARNING: num_pids_per_batch was 0. "
                    f"Adjusted to {self.num_pids_per_batch} "
                    f"(mini_batch_size adjusted to {self.mini_batch_size})"
                )

        # 디버깅: batch size 설정 확인
        if dist.get_rank() == 0:
            print(f"StratifiedIdentitySampler_DDP initialization:")
            print(f"  - batch_size (ims_per_batch): {self.batch_size}")
            print(f"  - world_size: {self.world_size}")
            print(f"  - mini_batch_size per GPU: {self.mini_batch_size}")
            print(f"  - num_instances: {self.num_instances}")
            print(f"  - num_pids_per_batch: {self.num_pids_per_batch}")
            print(
                f"  - Effective batch size per GPU: {self.num_pids_per_batch * self.num_instances}"
            )

        # 층 정보가 있는지 확인
        self.has_floor_info = False
        if len(data_source) > 0:
            first_item = data_source[0]
            if len(first_item) == 5:
                self.has_floor_info = True

        self.index_dic = defaultdict(list)
        for index, item in enumerate(self.data_source):
            pid = item[1]
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

        self.rank = dist.get_rank()
        self.length //= self.world_size

    def _sample_with_floor(self, pid, idxs):
        """층 정보를 고려하여 샘플링"""
        floor_dict = defaultdict(list)
        for idx in idxs:
            item = self.data_source[idx]
            img_path = item[0]
            floor = extract_floor_from_path(img_path)
            if floor is None:
                return None
            floor_dict[floor].append(idx)

        if len(floor_dict) < self.num_instances:
            return None

        # 층별로 셔플
        for floor in floor_dict:
            np.random.shuffle(floor_dict[floor])

        # 층별로 순환하며 샘플링
        batch_list = []
        floors = list(floor_dict.keys())
        floor_indices = {floor: 0 for floor in floors}

        # 충분한 배치 생성
        max_batches = max(len(floor_dict[floor]) for floor in floors)
        for _ in range(max_batches):
            batch = []
            for floor in floors:
                if len(batch) >= self.num_instances:
                    break
                if floor_indices[floor] < len(floor_dict[floor]):
                    batch.append(floor_dict[floor][floor_indices[floor]])
                    floor_indices[floor] += 1
                else:
                    batch.append(np.random.choice(floor_dict[floor]))

            if len(batch) == self.num_instances:
                batch_list.append(batch)
            else:
                break

        return batch_list

    def _sample_without_floor(self, pid, idxs):
        """층 정보 없이 기존 방식으로 샘플링 (fallback)"""
        idxs = copy.deepcopy(idxs)
        if len(idxs) < self.num_instances:
            idxs = np.random.choice(
                idxs, size=self.num_instances, replace=True
            ).tolist()
        np.random.shuffle(idxs)

        batch_list = []
        batch = []
        for idx in idxs:
            batch.append(idx)
            if len(batch) == self.num_instances:
                batch_list.append(batch)
                batch = []

        if len(batch) > 0:
            while len(batch) < self.num_instances:
                batch.append(np.random.choice(idxs))
            batch_list.append(batch)

        return batch_list

    def __iter__(self):
        """
        Lightning 표준: Sampler는 인덱스를 하나씩 반환해야 함.
        BatchSampler가 이를 mini_batch_size 개씩 묶어서 배치로 만듦.
        """
        seed = shared_random_seed()
        np.random.seed(seed)
        self._seed = int(seed)
        final_idxs = self.sample_list()
        length = int(math.ceil(len(final_idxs) * 1.0 / self.world_size))
        final_idxs = self.__fetch_current_node_idxs(final_idxs, length)
        self.length = len(final_idxs)
        # __len__에서 사용할 수 있도록 캐시 업데이트
        self._cached_length = self.length

        # 디버깅: 인덱스 리스트 확인 (첫 번째 epoch만)
        if dist.get_rank() == 0 and not hasattr(self, "_debug_printed"):
            self._debug_printed = True
            print(
                f"DEBUG: StratifiedIdentitySampler_DDP.__iter__ - final_idxs length: {len(final_idxs)}"
            )
            print(
                f"DEBUG: StratifiedIdentitySampler_DDP.__iter__ - first 10 indices: {final_idxs[:10] if len(final_idxs) >= 10 else final_idxs}"
            )
            print(
                f"DEBUG: StratifiedIdentitySampler_DDP.__iter__ - mini_batch_size: {self.mini_batch_size}"
            )

        # Lightning 표준: 인덱스를 하나씩 반환 (BatchSampler가 묶음)
        for idx in final_idxs:
            yield idx

    def __fetch_current_node_idxs(self, final_idxs, length):
        """
        각 rank에 대해 연속된 인덱스 블록을 할당.
        BatchSampler가 제대로 작동하도록 인덱스 리스트를 반환.
        """
        total_num = len(final_idxs)
        # 각 rank가 받을 인덱스 범위 계산
        start_idx = self.rank * length
        end_idx = min(start_idx + length, total_num)

        # 연속된 인덱스 블록 반환 (BatchSampler가 이를 mini_batch_size 개씩 묶음)
        rank_idxs = final_idxs[start_idx:end_idx]
        return rank_idxs

    def sample_list(self):
        avai_pids = copy.deepcopy(self.pids)
        batch_idxs_dict = {}
        batch_indices = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = np.random.choice(
                avai_pids, self.num_pids_per_batch, replace=False
            ).tolist()
            for pid in selected_pids:
                if pid not in batch_idxs_dict:
                    idxs = copy.deepcopy(self.index_dic[pid])
                    if self.has_floor_info:
                        batch_list = self._sample_with_floor(pid, idxs)
                        if batch_list is None:
                            batch_list = self._sample_without_floor(pid, idxs)
                    else:
                        batch_list = self._sample_without_floor(pid, idxs)
                    batch_idxs_dict[pid] = batch_list

                if len(batch_idxs_dict[pid]) > 0:
                    batch = batch_idxs_dict[pid].pop(0)
                    batch_indices.extend(batch)
                    if len(batch_idxs_dict[pid]) == 0:
                        avai_pids.remove(pid)
                else:
                    avai_pids.remove(pid)

        # 남은 identity 처리
        while len(avai_pids) > 0:
            num_available = min(len(avai_pids), self.num_pids_per_batch)
            if num_available == 0:
                break
            selected_pids = np.random.choice(
                avai_pids, num_available, replace=False
            ).tolist()
            temp_batch = []
            for pid in selected_pids:
                if pid not in batch_idxs_dict:
                    idxs = copy.deepcopy(self.index_dic[pid])
                    if self.has_floor_info:
                        batch_list = self._sample_with_floor(pid, idxs)
                        if batch_list is None:
                            batch_list = self._sample_without_floor(pid, idxs)
                    else:
                        batch_list = self._sample_without_floor(pid, idxs)
                    batch_idxs_dict[pid] = batch_list

                if len(batch_idxs_dict[pid]) > 0:
                    batch = batch_idxs_dict[pid].pop(0)
                    temp_batch.extend(batch)
                    if len(batch_idxs_dict[pid]) == 0:
                        avai_pids.remove(pid)
                else:
                    avai_pids.remove(pid)

            if len(temp_batch) == self.mini_batch_size:
                batch_indices.extend(temp_batch)

        return batch_indices

    def __len__(self):
        """
        Lightning 표준: Sampler는 인덱스 개수를 반환해야 함.
        BatchSampler가 배치 개수를 계산함.

        주의: __iter__에서 sample_list()를 호출하여 실제 인덱스 리스트를 생성하므로,
        __len__에서도 동일한 방식으로 계산해야 정확한 길이를 반환할 수 있습니다.
        하지만 sample_list()는 랜덤 샘플링을 수행하므로, 매번 다른 결과를 반환할 수 있습니다.

        해결책: sample_list()를 호출하여 실제 생성될 인덱스 개수를 계산합니다.
        계산 결과를 캐싱하여 오버헤드를 줄입니다.
        """
        # 캐시된 값이 있으면 사용
        if self._cached_length is not None:
            return self._cached_length

        # sample_list()를 호출하여 실제 생성될 인덱스 개수 계산
        # seed를 고정하여 일관된 결과 보장
        original_seed = random.getstate() if hasattr(random, "getstate") else None
        np_state = np.random.get_state()

        try:
            # 고정된 seed로 샘플링하여 일관된 길이 계산
            test_seed = 42  # 고정된 seed
            np.random.seed(test_seed)
            final_idxs = self.sample_list()
            length = int(math.ceil(len(final_idxs) * 1.0 / self.world_size))
            final_idxs = self.__fetch_current_node_idxs(final_idxs, length)
            self._cached_length = len(final_idxs)
            return self._cached_length
        finally:
            # random state 복원
            if original_seed is not None:
                random.setstate(original_seed)
            np.random.set_state(np_state)
