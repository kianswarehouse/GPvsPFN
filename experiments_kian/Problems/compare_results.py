import json
import os
from pathlib import Path

def compare_folders(base_path, folder1, folder2, problem_name):
    """Compare two result folders for a given problem"""
    print(f"\n{'='*80}")
    print(f"Comparing {problem_name}: {folder1} vs {folder2}")
    print(f"{'='*80}\n")
    
    path1 = Path(base_path) / problem_name / folder1 / "default"
    path2 = Path(base_path) / problem_name / folder2 / "default"
    
    # Get all JSON files
    files1 = set(f.name for f in path1.glob("*.json")) if path1.exists() else set()
    files2 = set(f.name for f in path2.glob("*.json")) if path2.exists() else set()
    
    # Files only in folder1
    only_in_1 = files1 - files2
    # Files only in folder2
    only_in_2 = files2 - files1
    # Files in both
    in_both = files1 & files2
    
    print(f"Files only in {folder1}: {len(only_in_1)}")
    for f in sorted(only_in_1):
        print(f"  - {f}")
    
    print(f"\nFiles only in {folder2}: {len(only_in_2)}")
    for f in sorted(only_in_2):
        print(f"  - {f}")
    
    print(f"\nFiles in both folders: {len(in_both)}")
    
    # Compare content of files in both
    differences = []
    for filename in sorted(in_both):
        file1 = path1 / filename
        file2 = path2 / filename
        
        try:
            with open(file1, 'r') as f:
                data1 = json.load(f)
            with open(file2, 'r') as f:
                data2 = json.load(f)
            
            # Check if identical
            if data1 != data2:
                differences.append(filename)
                # Check key differences
                print(f"\n  Differences in {filename}:")
                
                # Check count differences
                if 'gp_data' in data1 and 'gp_data' in data2:
                    count1 = data1['gp_data']['summary'].get('RRMSE', {}).get('count', 'N/A')
                    count2 = data2['gp_data']['summary'].get('RRMSE', {}).get('count', 'N/A')
                    if count1 != count2:
                        print(f"    - GP RRMSE count: {count1} vs {count2}")
                    
                    mean1 = data1['gp_data']['summary'].get('RRMSE', {}).get('mean', 'N/A')
                    mean2 = data2['gp_data']['summary'].get('RRMSE', {}).get('mean', 'N/A')
                    if mean1 != mean2:
                        print(f"    - GP RRMSE mean: {mean1:.6f} vs {mean2:.6f}")
                    
                    # Check S0 (high-fidelity) metrics for multi-fidelity problems
                    per_source1 = data1['gp_data']['summary'].get('per_source', {})
                    per_source2 = data2['gp_data']['summary'].get('per_source', {})
                    if 'source_0' in per_source1 and 'source_0' in per_source2:
                        s0_rrmse1 = per_source1['source_0'].get('RRMSE', {}).get('mean', 'N/A')
                        s0_rrmse2 = per_source2['source_0'].get('RRMSE', {}).get('mean', 'N/A')
                        s0_nis1 = per_source1['source_0'].get('NIS', {}).get('mean', 'N/A')
                        s0_nis2 = per_source2['source_0'].get('NIS', {}).get('mean', 'N/A')
                        
                        if s0_rrmse1 != 'N/A' and s0_rrmse2 != 'N/A' and s0_rrmse1 != s0_rrmse2:
                            diff_pct = ((s0_rrmse2 - s0_rrmse1) / s0_rrmse1) * 100 if s0_rrmse1 != 0 else 0
                            print(f"    - GP S0 (high-fidelity) RRMSE: {s0_rrmse1:.6f} vs {s0_rrmse2:.6f} ({diff_pct:+.1f}%)")
                        
                        if s0_nis1 != 'N/A' and s0_nis2 != 'N/A' and s0_nis1 != s0_nis2:
                            diff_pct = ((s0_nis2 - s0_nis1) / s0_nis1) * 100 if s0_nis1 != 0 else 0
                            print(f"    - GP S0 (high-fidelity) NIS: {s0_nis1:.6f} vs {s0_nis2:.6f} ({diff_pct:+.1f}%)")
                
                if 'tabpfn_data' in data1 and 'tabpfn_data' in data2:
                    count1 = data1['tabpfn_data']['summary'].get('RRMSE', {}).get('count', 'N/A')
                    count2 = data2['tabpfn_data']['summary'].get('RRMSE', {}).get('count', 'N/A')
                    if count1 != count2:
                        print(f"    - TabPFN RRMSE count: {count1} vs {count2}")
                    
                    mean1 = data1['tabpfn_data']['summary'].get('RRMSE', {}).get('mean', 'N/A')
                    mean2 = data2['tabpfn_data']['summary'].get('RRMSE', {}).get('mean', 'N/A')
                    if mean1 != mean2:
                        print(f"    - TabPFN RRMSE mean: {mean1:.6f} vs {mean2:.6f}")
                    
                    # Check S0 (high-fidelity) metrics for multi-fidelity problems
                    per_source1 = data1['tabpfn_data']['summary'].get('per_source', {})
                    per_source2 = data2['tabpfn_data']['summary'].get('per_source', {})
                    if 'source_0' in per_source1 and 'source_0' in per_source2:
                        s0_rrmse1 = per_source1['source_0'].get('RRMSE', {}).get('mean', 'N/A')
                        s0_rrmse2 = per_source2['source_0'].get('RRMSE', {}).get('mean', 'N/A')
                        s0_nis1 = per_source1['source_0'].get('NIS', {}).get('mean', 'N/A')
                        s0_nis2 = per_source2['source_0'].get('NIS', {}).get('mean', 'N/A')
                        
                        if s0_rrmse1 != 'N/A' and s0_rrmse2 != 'N/A' and s0_rrmse1 != s0_rrmse2:
                            diff_pct = ((s0_rrmse2 - s0_rrmse1) / s0_rrmse1) * 100 if s0_rrmse1 != 0 else 0
                            print(f"    - TabPFN S0 (high-fidelity) RRMSE: {s0_rrmse1:.6f} vs {s0_rrmse2:.6f} ({diff_pct:+.1f}%)")
                        
                        if s0_nis1 != 'N/A' and s0_nis2 != 'N/A' and s0_nis1 != s0_nis2:
                            diff_pct = ((s0_nis2 - s0_nis1) / s0_nis1) * 100 if s0_nis1 != 0 else 0
                            print(f"    - TabPFN S0 (high-fidelity) NIS: {s0_nis1:.6f} vs {s0_nis2:.6f} ({diff_pct:+.1f}%)")
        except Exception as e:
            print(f"    Error comparing {filename}: {e}")
    
    if not differences:
        print("  All matching files have identical content.")
    
    # Check noise folders if they exist
    noise_path1 = Path(base_path) / problem_name / folder1 / "noise"
    noise_path2 = Path(base_path) / problem_name / folder2 / "noise"
    
    if noise_path1.exists() or noise_path2.exists():
        print(f"\nNoise folder comparison:")
        noise_files1 = set(f.name for f in noise_path1.glob("*.json")) if noise_path1.exists() else set()
        noise_files2 = set(f.name for f in noise_path2.glob("*.json")) if noise_path2.exists() else set()
        
        only_noise_1 = noise_files1 - noise_files2
        only_noise_2 = noise_files2 - noise_files1
        
        if only_noise_1:
            print(f"  Files only in {folder1}/noise: {len(only_noise_1)}")
            for f in sorted(only_noise_1):
                print(f"    - {f}")
        
        if only_noise_2:
            print(f"  Files only in {folder2}/noise: {len(only_noise_2)}")
            for f in sorted(only_noise_2):
                print(f"    - {f}")
        
        if not only_noise_1 and not only_noise_2:
            print(f"  Noise folders have same files")

if __name__ == "__main__":
    base_path = Path("results")
    folder1 = "11_17"
    folder2 = "11_172"
    
    problems = ["wing", "buckling", "borehole"]
    
    for problem in problems:
        compare_folders(base_path, folder1, folder2, problem)

