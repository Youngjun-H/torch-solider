#!/usr/bin/env python3
"""
Simple script to check CCTV ReID dataset structure without dependencies
"""

import os
import glob
import re
from pathlib import Path
from collections import Counter


def check_cctv_dataset():
    """Check CCTV ReID dataset structure"""

    dataset_root = "/purestorage/AILAB/AI_2/datasets/PersonReID/cctv_reid_dataset_v2"

    print("=" * 80)
    print("CCTV ReID Dataset Structure Check")
    print("=" * 80)
    print(f"\nDataset root: {dataset_root}")

    # Check directories exist
    train_dir = os.path.join(dataset_root, 'train')
    query_dir = os.path.join(dataset_root, 'valid', 'query')
    gallery_dir = os.path.join(dataset_root, 'valid', 'gallery')

    for dir_name, dir_path in [('Train', train_dir), ('Query', query_dir), ('Gallery', gallery_dir)]:
        if os.path.exists(dir_path):
            print(f"✅ {dir_name} directory exists: {dir_path}")
        else:
            print(f"❌ {dir_name} directory NOT found: {dir_path}")
            return False

    # Count identities and images
    print("\n" + "=" * 80)
    print("Dataset Statistics:")
    print("=" * 80)

    # Training set
    train_persons = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    train_image_count = sum(len(glob.glob(os.path.join(train_dir, p, '*.jpg'))) for p in train_persons)

    print(f"\nTraining Set:")
    print(f"  Number of identities: {len(train_persons)}")
    print(f"  Number of images: {train_image_count}")
    print(f"  Average images per identity: {train_image_count / len(train_persons):.1f}")

    # Query set
    query_persons = [d for d in os.listdir(query_dir) if os.path.isdir(os.path.join(query_dir, d))]
    query_image_count = sum(len(glob.glob(os.path.join(query_dir, p, '*.jpg'))) for p in query_persons)

    print(f"\nQuery Set:")
    print(f"  Number of identities: {len(query_persons)}")
    print(f"  Number of images: {query_image_count}")
    print(f"  Average images per identity: {query_image_count / len(query_persons):.1f}")

    # Gallery set
    gallery_persons = [d for d in os.listdir(gallery_dir) if os.path.isdir(os.path.join(gallery_dir, d))]
    gallery_image_count = sum(len(glob.glob(os.path.join(gallery_dir, p, '*.jpg'))) for p in gallery_persons)

    print(f"\nGallery Set:")
    print(f"  Number of identities: {len(gallery_persons)}")
    print(f"  Number of images: {gallery_image_count}")
    print(f"  Average images per identity: {gallery_image_count / len(gallery_persons):.1f}")

    # Sample images from training
    print("\n" + "=" * 80)
    print("Sample Training Images (first 5 identities, 3 images each):")
    print("=" * 80)

    for i, person_id in enumerate(sorted(train_persons)[:5]):
        person_path = os.path.join(train_dir, person_id)
        img_files = sorted(glob.glob(os.path.join(person_path, '*.jpg')))[:3]
        print(f"\n{i+1}. Identity: {person_id} (Total: {len(glob.glob(os.path.join(person_path, '*.jpg')))} images)")
        for img_file in img_files:
            print(f"   - {Path(img_file).name}")

    # Camera distribution analysis
    print("\n" + "=" * 80)
    print("Camera Distribution Analysis:")
    print("=" * 80)

    def extract_camera(filename):
        pattern = r'\d{4}-\d{2}-\d{2}_([^_]+)_\d{2}-\d{2}-\d{2}'
        match = re.search(pattern, filename)
        return match.group(1) if match else 'Unknown'

    # Analyze training set cameras
    train_cameras = []
    for person_id in train_persons:
        person_path = os.path.join(train_dir, person_id)
        for img_file in glob.glob(os.path.join(person_path, '*.jpg')):
            camera = extract_camera(Path(img_file).name)
            train_cameras.append(camera)

    camera_counts = Counter(train_cameras)
    print(f"\nTraining Set Cameras:")
    for camera, count in sorted(camera_counts.items()):
        print(f"  {camera}: {count} images ({count/len(train_cameras)*100:.1f}%)")

    # Verify query/gallery overlap
    query_ids = set([p.replace('ID', '') for p in query_persons if 'ID' in p])
    gallery_ids = set([p.replace('ID', '') for p in gallery_persons if 'ID' in p])
    overlap = query_ids.intersection(gallery_ids)

    print(f"\n" + "=" * 80)
    print("Query/Gallery Overlap Check:")
    print("=" * 80)
    print(f"Query IDs: {len(query_ids)}")
    print(f"Gallery IDs: {len(gallery_ids)}")
    print(f"Overlapping IDs: {len(overlap)} ({'✅ Good' if len(overlap) == len(query_ids) else '⚠️  Warning'})")

    print("\n" + "=" * 80)
    print("✅ Dataset structure verification complete!")
    print("=" * 80)

    return True


if __name__ == '__main__':
    import sys
    success = check_cctv_dataset()
    sys.exit(0 if success else 1)
