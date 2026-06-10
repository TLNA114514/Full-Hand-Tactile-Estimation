import json
import subprocess
import sys

# We'll just patch eval_tactile.py temporarily to print the exception, or run it and capture.
# Actually, let's just modify eval_tactile.py to print the exception instead of silently continuing!
with open('/code/users/jiangrui/Full-Hand-Tactile-Estimation/hamer_tactile_ft/eval_tactile.py', 'r') as f:
    code = f.read()

code = code.replace(
    '''            except Exception as e:
                continue''',
    '''            except Exception as e:
                print(f"Exception in forward_step: {e}")
                continue'''
)

with open('/code/users/jiangrui/Full-Hand-Tactile-Estimation/hamer_tactile_ft/eval_tactile.py', 'w') as f:
    f.write(code)

print("eval_tactile.py modified to print exceptions.")
