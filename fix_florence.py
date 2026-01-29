import os

def fix_all_florence_bugs_v3():
    target_files = [
        "./models/local_florence2/modeling_florence2.py",
        os.path.expanduser("~/.cache/huggingface/modules/transformers_modules/local_florence2/modeling_florence2.py").replace("/", "\\")
    ]
    
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

        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        new_lines = []
        file_fixed_count = 0
        
        for line in lines:
            stripped = line.strip()
            indent = line[:len(line) - len(line.lstrip())]
            
            if "past_key_values[0][0].shape[2]" in line and "is not None" not in line:
                var_name = line.split('=')[0].strip()
                safe_check = "past_key_values[0][0].shape[2] if past_key_values is not None and past_key_values[0] is not None and past_key_values[0][0] is not None else 0"
                new_line = f"{indent}{var_name} = {safe_check}\n"
                new_lines.append(new_line)
                file_fixed_count += 1
            
            elif "torch.cat([past_key_value[0], key_states], dim=2)" in line and "if past_key_value[0] is not None" not in line:
                new_line = f"{indent}key_states = torch.cat([past_key_value[0], key_states], dim=2) if past_key_value[0] is not None else key_states\n"
                new_lines.append(new_line)
                file_fixed_count += 1

            elif "torch.cat([past_key_value[1], value_states], dim=2)" in line and "if past_key_value[1] is not None" not in line:
                new_line = f"{indent}value_states = torch.cat([past_key_value[1], value_states], dim=2) if past_key_value[1] is not None else value_states\n"
                new_lines.append(new_line)
                file_fixed_count += 1

            elif "past_key_value[0].shape[2] == key_value_states.shape[1]" in line and "past_key_value[0] is not None" not in line:
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
                print(f" fix file: {file_fixed_count} 。")
                fixed_any = True
            except PermissionError:
                print(f"fail: {file_path}")
        else:
            print("No bug")

    if fixed_any:
        for file_path in target_files:
            if os.path.exists(file_path):
                cache_dir = os.path.join(os.path.dirname(file_path), "__pycache__")
                if os.path.exists(cache_dir):
                    import shutil
                    try:
                        shutil.rmtree(cache_dir)
                    except:
                        pass
        print("\nfinished")

if __name__ == "__main__":
    fix_all_florence_bugs_v3()
