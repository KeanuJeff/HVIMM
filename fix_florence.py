import os

def fix_all_florence_bugs_v3():
    # 設定目標檔案路徑 (包含本地和緩存)
    target_files = [
        "./models/local_florence2/modeling_florence2.py",
        os.path.expanduser("~/.cache/huggingface/modules/transformers_modules/local_florence2/modeling_florence2.py").replace("/", "\\")
    ]
    
    # Windows 使用者路徑檢查
    if os.name == 'nt':
        user_profile = os.environ.get('USERPROFILE')
        if user_profile:
            cache_path = os.path.join(user_profile, ".cache", "huggingface", "modules", "transformers_modules", "local_florence2", "modeling_florence2.py")
            if cache_path not in target_files:
                target_files.append(cache_path)

    fixed_any = False

    for file_path in target_files:
        if not os.path.exists(file_path):
            continue

        print(f"\n正在掃描檔案: {file_path} ...")
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        new_lines = []
        file_fixed_count = 0
        
        for line in lines:
            stripped = line.strip()
            indent = line[:len(line) - len(line.lstrip())]
            
            # --- Bug 1: shape[2] 取長度錯誤 (prepare_inputs_for_generation) ---
            if "past_key_values[0][0].shape[2]" in line and "is not None" not in line:
                print(f"   🛠️  修復 Bug 1: shape[2] 讀取錯誤 (Line {lines.index(line) + 1})")
                var_name = line.split('=')[0].strip()
                safe_check = "past_key_values[0][0].shape[2] if past_key_values is not None and past_key_values[0] is not None and past_key_values[0][0] is not None else 0"
                new_line = f"{indent}{var_name} = {safe_check}\n"
                new_lines.append(new_line)
                file_fixed_count += 1
            
            # --- Bug 2: torch.cat Key 錯誤 ---
            elif "torch.cat([past_key_value[0], key_states], dim=2)" in line and "if past_key_value[0] is not None" not in line:
                print(f"   🛠️  修復 Bug 2: torch.cat Key 錯誤 (Line {lines.index(line) + 1})")
                new_line = f"{indent}key_states = torch.cat([past_key_value[0], key_states], dim=2) if past_key_value[0] is not None else key_states\n"
                new_lines.append(new_line)
                file_fixed_count += 1

            # --- Bug 3: torch.cat Value 錯誤 ---
            elif "torch.cat([past_key_value[1], value_states], dim=2)" in line and "if past_key_value[1] is not None" not in line:
                print(f"   🛠️  修復 Bug 3: torch.cat Value 錯誤 (Line {lines.index(line) + 1})")
                new_line = f"{indent}value_states = torch.cat([past_key_value[1], value_states], dim=2) if past_key_value[1] is not None else value_states\n"
                new_lines.append(new_line)
                file_fixed_count += 1

            # --- Bug 4 (NEW): Attention shape check 錯誤 ---
            # 原始: and past_key_value[0].shape[2] == key_value_states.shape[1]
            elif "past_key_value[0].shape[2] == key_value_states.shape[1]" in line and "past_key_value[0] is not None" not in line:
                print(f"   🛠️  修復 Bug 4: Attention shape 檢查錯誤 (Line {lines.index(line) + 1})")
                # 我們在判斷式中插入 "and past_key_value[0] is not None"
                # 這裡使用 replace 比較安全，因為它通常在 if 語句中間
                new_line = line.replace(
                    "past_key_value[0].shape[2]", 
                    "past_key_value[0] is not None and past_key_value[0].shape[2]"
                )
                new_lines.append(new_line)
                file_fixed_count += 1
            
            else:
                new_lines.append(line)

        if file_fixed_count > 0:
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.writelines(new_lines)
                print(f"✅ 已寫入檔案！共修復 {file_fixed_count} 處。")
                fixed_any = True
            except PermissionError:
                print(f"❌ 無法寫入檔案 (權限不足): {file_path}")
        else:
            print("✨ 此檔案看起來已經修復或沒有發現 Bug。")

    if fixed_any:
        print("\n🧹 正在清除 __pycache__ ...")
        for file_path in target_files:
            if os.path.exists(file_path):
                cache_dir = os.path.join(os.path.dirname(file_path), "__pycache__")
                if os.path.exists(cache_dir):
                    import shutil
                    try:
                        shutil.rmtree(cache_dir)
                    except:
                        pass
        print("\n🎉 修復完成！請重新執行 preprocess_refcocog.py")

if __name__ == "__main__":
    fix_all_florence_bugs_v3()