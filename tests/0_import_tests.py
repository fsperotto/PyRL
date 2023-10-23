print("*****************")
print("pytorch")
print("*****************")
try:
    import torch
    print(torch.rand(5, 3))
except Exception as e:
    print("An exception occurred while importing TORCH:", e)    
else:
    print('OK.')
print()

print("*****************")
print("tf")
print("*****************")
try:
    import tensorflow as tf
    print(tf.reduce_sum(tf.random.normal([1000, 1000])))
except Exception as e:
    print("An exception occurred while importing TF:", e)    
else:
    print('OK.')
print()
print()

print("*****************")
print("sb3")
print("*****************")
try:
    import stable_baselines3 as sb3
except Exception as e:
    print("An exception occurred while importing SB3:", e)    
else:
    print('OK.')
print()
print()