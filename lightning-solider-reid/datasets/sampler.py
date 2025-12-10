import copy
import os.path as osp
import random
import re
from collections import defaultdict

import numpy as np
import torch
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


class RandomIdentitySampler(Sampler):
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
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)  # dict with list value
        # {783: [0, 5, 116, 876, 1554, 2041],...,}
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

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

            # 남은 인덱스 처리: 부족하면 복원 샘플링으로 채움
            if len(batch_idxs) > 0:
                while len(batch_idxs) < self.num_instances:
                    batch_idxs.append(np.random.choice(idxs))
                batch_idxs_dict[pid].append(batch_idxs)

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                if len(batch_idxs_dict[pid]) > 0:  # 안전성 체크
                    batch_idxs = batch_idxs_dict[pid].pop(0)
                    final_idxs.extend(batch_idxs)
                    if len(batch_idxs_dict[pid]) == 0:
                        avai_pids.remove(pid)

        # 남은 identity 처리: 가능한 만큼 배치 구성
        while len(avai_pids) > 0:
            # 남은 identity가 num_pids_per_batch보다 적으면, 가능한 만큼만 사용
            num_available = min(len(avai_pids), self.num_pids_per_batch)
            if num_available == 0:
                break
            selected_pids = random.sample(avai_pids, num_available)
            temp_batch = []
            for pid in selected_pids:
                if len(batch_idxs_dict[pid]) > 0:
                    batch_idxs = batch_idxs_dict[pid].pop(0)
                    temp_batch.extend(batch_idxs)
                    if len(batch_idxs_dict[pid]) == 0:
                        avai_pids.remove(pid)

            # 배치 크기가 맞으면 추가, 아니면 버림 (마지막 불완전 배치)
            if len(temp_batch) == self.batch_size:
                final_idxs.extend(temp_batch)

        return iter(final_idxs)

    def __len__(self):
        return self.length


# New add by gu
class RandomIdentitySampler_IdUniform(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.

    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/data/sampler.py.

    Args:
        data_source (Dataset): dataset to sample from.
        num_instances (int): number of instances per identity.
    """

    def __init__(self, data_source, num_instances):
        self.data_source = data_source
        self.num_instances = num_instances
        self.index_dic = defaultdict(list)
        for index, item in enumerate(data_source):
            pid = item[1]
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())
        self.num_identities = len(self.pids)

    def __iter__(self):
        indices = torch.randperm(self.num_identities)
        ret = []
        for i in indices:
            pid = self.pids[i]
            t = self.index_dic[pid]
            replace = False if len(t) >= self.num_instances else True
            t = np.random.choice(t, size=self.num_instances, replace=replace)
            ret.extend(t)
        return iter(ret)

    def __len__(self):
        return self.num_identities * self.num_instances


class StratifiedIdentitySampler(Sampler):
    """
    층 정보를 고려한 PK 샘플러

    각 identity에서 K개 샘플링 시, 층 정보가 있으면 각 층에서 균등하게 샘플링.
    층 정보가 없으면 기존 RandomIdentitySampler처럼 랜덤 샘플링 (fallback).

    Args:
        data_source (list): list of (img_path, pid, camid, trackid) or
                           (img_path, pid, camid, trackid, floor)
        batch_size (int): number of examples in a batch
        num_instances (int): number of instances per identity in a batch
    """

    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances

        # Identity별 인덱스 그룹화
        self.index_dic = defaultdict(list)
        # 층 정보가 있는지 확인
        self.has_floor_info = False
        if len(data_source) > 0:
            # 첫 번째 샘플에서 층 정보 확인
            first_item = data_source[0]
            if len(first_item) == 5:
                self.has_floor_info = True

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

    def _sample_with_floor(self, pid, idxs):
        """
        층 정보를 고려하여 샘플링

        Args:
            pid: identity ID
            idxs: 해당 identity의 인덱스 리스트

        Returns:
            list: 샘플링된 인덱스 리스트 (K개씩 묶인 배치들)
        """
        # 층별로 그룹화
        floor_dict = defaultdict(list)
        for idx in idxs:
            item = self.data_source[idx]
            img_path = item[0]
            floor = extract_floor_from_path(img_path)
            if floor is None:
                # 층 정보가 없으면 fallback
                return None
            floor_dict[floor].append(idx)

        # 층이 충분하지 않으면 fallback
        if len(floor_dict) < self.num_instances:
            return None

        # 각 층에서 셔플
        floors = list(floor_dict.keys())
        for floor in floors:
            random.shuffle(floor_dict[floor])

        batch_idxs_list = []
        floor_indices = {floor: 0 for floor in floors}

        # 가능한 모든 배치 생성
        while True:
            batch_idxs = []
            # 각 층에서 하나씩 샘플링 (round-robin)
            for floor in floors:
                if len(batch_idxs) >= self.num_instances:
                    break
                if floor_indices[floor] < len(floor_dict[floor]):
                    batch_idxs.append(floor_dict[floor][floor_indices[floor]])
                    floor_indices[floor] += 1
                else:
                    # 해당 층의 이미지가 부족하면 복원 샘플링
                    batch_idxs.append(np.random.choice(floor_dict[floor]))

            if len(batch_idxs) == self.num_instances:
                batch_idxs_list.append(batch_idxs)
            else:
                # 더 이상 배치를 만들 수 없으면 종료
                break

        # 남은 이미지들로 추가 배치 구성 (층 정보 없이)
        remaining_idxs = []
        for floor in floors:
            start_idx = floor_indices[floor]
            remaining_idxs.extend(floor_dict[floor][start_idx:])

        # 남은 이미지가 충분하면 추가 배치 구성
        if len(remaining_idxs) >= self.num_instances:
            random.shuffle(remaining_idxs)
            while len(remaining_idxs) >= self.num_instances:
                batch_idxs_list.append(remaining_idxs[: self.num_instances])
                remaining_idxs = remaining_idxs[self.num_instances :]

        return batch_idxs_list if batch_idxs_list else None

    def _sample_without_floor(self, pid, idxs):
        """
        층 정보 없이 기존 방식으로 샘플링 (fallback)

        Args:
            pid: identity ID
            idxs: 해당 identity의 인덱스 리스트

        Returns:
            list: 샘플링된 인덱스 리스트 (K개씩 묶인 배치들)
        """
        idxs = copy.deepcopy(idxs)
        if len(idxs) < self.num_instances:
            idxs = np.random.choice(
                idxs, size=self.num_instances, replace=True
            ).tolist()
        random.shuffle(idxs)

        batch_idxs_list = []
        batch_idxs = []
        for idx in idxs:
            batch_idxs.append(idx)
            if len(batch_idxs) == self.num_instances:
                batch_idxs_list.append(batch_idxs)
                batch_idxs = []

        # 남은 인덱스 처리
        if len(batch_idxs) > 0:
            while len(batch_idxs) < self.num_instances:
                batch_idxs.append(np.random.choice(idxs))
            batch_idxs_list.append(batch_idxs)

        return batch_idxs_list

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = self.index_dic[pid]

            # 층 정보가 있으면 층별 샘플링, 없으면 기존 방식
            if self.has_floor_info:
                batch_list = self._sample_with_floor(pid, idxs)
                if batch_list is None:
                    # 층 정보가 부족하면 fallback
                    batch_list = self._sample_without_floor(pid, idxs)
            else:
                batch_list = self._sample_without_floor(pid, idxs)

            batch_idxs_dict[pid] = batch_list

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                if len(batch_idxs_dict[pid]) > 0:
                    batch_idxs = batch_idxs_dict[pid].pop(0)
                    final_idxs.extend(batch_idxs)
                    if len(batch_idxs_dict[pid]) == 0:
                        avai_pids.remove(pid)

        # 남은 identity 처리
        while len(avai_pids) > 0:
            num_available = min(len(avai_pids), self.num_pids_per_batch)
            if num_available == 0:
                break
            selected_pids = random.sample(avai_pids, num_available)
            temp_batch = []
            for pid in selected_pids:
                if len(batch_idxs_dict[pid]) > 0:
                    batch_idxs = batch_idxs_dict[pid].pop(0)
                    temp_batch.extend(batch_idxs)
                    if len(batch_idxs_dict[pid]) == 0:
                        avai_pids.remove(pid)

            if len(temp_batch) == self.batch_size:
                final_idxs.extend(temp_batch)

        return iter(final_idxs)

    def __len__(self):
        return self.length
