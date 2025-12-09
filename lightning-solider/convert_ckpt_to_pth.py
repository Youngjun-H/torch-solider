"""SOLIDER Lightning checkpoint (.ckpt)를 PyTorch checkpoint (.pth)로 변환하는 스크립트."""

import argparse
import sys
from pathlib import Path

import torch

# lightning-solider 루트 디렉토리를 path에 추가
sys.path.insert(0, str(Path(__file__).parent))


def extract_state_dict(state_dict, prefix=""):
    """
    체크포인트에서 실제 state_dict를 추출합니다.
    Lightning 모듈 prefix를 제거합니다.
    
    Args:
        state_dict: 원본 state_dict
        prefix: 제거할 prefix (예: "student.", "teacher.")
    
    Returns:
        추출된 state_dict
    """
    if not isinstance(state_dict, dict):
        return state_dict
    
    result = {}
    for k, v in state_dict.items():
        # DDP wrapper 제거
        new_key = k.replace("module.", "") if k.startswith("module.") else k
        # Lightning 모듈 prefix 제거
        if prefix and new_key.startswith(prefix):
            new_key = new_key[len(prefix):]
        result[new_key] = v
    return result


def convert_ckpt_to_pth(ckpt_path, output_path=None, save_format="combined"):
    """
    Lightning checkpoint (.ckpt)를 PyTorch checkpoint (.pth)로 변환합니다.
    
    Args:
        ckpt_path: 입력 .ckpt 파일 경로
        output_path: 출력 .pth 파일 경로 (None이면 자동 생성)
        save_format: 저장 형식
            - "combined": student, teacher, part_classifier를 하나의 dict로 저장
            - "separate": student, teacher, part_classifier를 각각 별도 파일로 저장
            - "student_only": student만 저장
            - "teacher_only": teacher만 저장
    """
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")
    
    print(f"Loading checkpoint from {ckpt_path}...")
    
    # Load checkpoint
    # weights_only=False: PyTorch 2.6+에서 numpy 객체를 포함한 체크포인트 로드를 위해 필요
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    
    # Lightning checkpoint 형식인지 확인
    is_lightning_checkpoint = (
        "state_dict" in checkpoint and "pytorch-lightning_version" in checkpoint
    )
    
    if not is_lightning_checkpoint:
        print("Warning: This doesn't appear to be a Lightning checkpoint.")
        print("Attempting to extract state_dict directly...")
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint["state_dict"]
        print(f"Lightning version: {checkpoint.get('pytorch-lightning_version', 'unknown')}")
        print(f"Epoch: {checkpoint.get('epoch', 'unknown')}")
        print(f"Global step: {checkpoint.get('global_step', 'unknown')}")
    
    # Student, Teacher, Part Classifier 분리
    student_state = {}
    teacher_state = {}
    part_classifier_state = {}
    other_state = {}
    
    for k, v in state_dict.items():
        if k.startswith("student."):
            student_state[k[7:]] = v  # "student." 제거 (7자)
        elif k.startswith("teacher."):
            teacher_state[k[8:]] = v  # "teacher." 제거 (8자)
        elif k.startswith("part_classifier."):
            part_classifier_state[k[16:]] = v  # "part_classifier." 제거 (16자)
        else:
            other_state[k] = v
    
    print(f"\nFound components:")
    print(f"  - Student: {len(student_state)} parameters")
    print(f"  - Teacher: {len(teacher_state)} parameters")
    print(f"  - Part Classifier: {len(part_classifier_state)} parameters")
    if other_state:
        print(f"  - Other: {len(other_state)} parameters")
    
    # 출력 경로 결정
    if output_path is None:
        output_path = ckpt_path.parent / f"{ckpt_path.stem}.pth"
    else:
        output_path = Path(output_path)
    
    # 저장 형식에 따라 저장
    if save_format == "combined":
        # 모든 컴포넌트를 하나의 dict로 저장
        pth_dict = {
            "student": student_state,
            "teacher": teacher_state,
            "part_classifier": part_classifier_state,
        }
        if other_state:
            pth_dict["other"] = other_state
        
        # 메타데이터 추가
        if is_lightning_checkpoint:
            pth_dict["epoch"] = checkpoint.get("epoch", None)
            pth_dict["global_step"] = checkpoint.get("global_step", None)
        
        torch.save(pth_dict, output_path)
        print(f"\nSaved combined checkpoint to: {output_path}")
        
    elif save_format == "separate":
        # 각 컴포넌트를 별도 파일로 저장
        base_path = output_path.parent / output_path.stem
        
        if student_state:
            student_path = base_path.parent / f"{base_path.name}_student.pth"
            torch.save(student_state, student_path)
            print(f"Saved student to: {student_path}")
        
        if teacher_state:
            teacher_path = base_path.parent / f"{base_path.name}_teacher.pth"
            torch.save(teacher_state, teacher_path)
            print(f"Saved teacher to: {teacher_path}")
        
        if part_classifier_state:
            part_path = base_path.parent / f"{base_path.name}_part_classifier.pth"
            torch.save(part_classifier_state, part_path)
            print(f"Saved part_classifier to: {part_path}")
    
    elif save_format == "student_only":
        if not student_state:
            raise ValueError("Student state_dict is empty!")
        torch.save(student_state, output_path)
        print(f"\nSaved student checkpoint to: {output_path}")
    
    elif save_format == "teacher_only":
        if not teacher_state:
            raise ValueError("Teacher state_dict is empty!")
        torch.save(teacher_state, output_path)
        print(f"\nSaved teacher checkpoint to: {output_path}")
    
    else:
        raise ValueError(f"Unknown save_format: {save_format}")
    
    print("\nConversion completed successfully!")


def main():
    parser = argparse.ArgumentParser(
        description="Convert SOLIDER Lightning checkpoint (.ckpt) to PyTorch checkpoint (.pth)"
    )
    parser.add_argument(
        "ckpt_path",
        type=str,
        help="Path to input .ckpt file",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Path to output .pth file (default: same name as input with .pth extension)",
    )
    parser.add_argument(
        "--format",
        "-f",
        type=str,
        default="combined",
        choices=["combined", "separate", "student_only", "teacher_only"],
        help="Save format: 'combined' (default), 'separate', 'student_only', or 'teacher_only'",
    )
    
    args = parser.parse_args()
    
    convert_ckpt_to_pth(args.ckpt_path, args.output, args.format)


if __name__ == "__main__":
    main()

