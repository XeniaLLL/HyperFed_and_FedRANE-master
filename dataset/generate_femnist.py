import json
import random
import numpy as np

dir_path = "femnist/"
train_path = dir_path + "train/"
test_path = dir_path + "test/"
train_data = []
test_data = []
chosen_user_list = []

# randomly choose 20 clients from 3500 users
for i in range(20):
    user_idx = random.randint(0, 35)  # first choose which file to read
    train_file_path = f"/home/zpy/Others/leaf/data/femnist/data/train/all_data_{user_idx}_niid_0_keep_0_train_75.json"
    test_file_path = f"/home/zpy/Others/leaf/data/femnist/data/test/all_data_{user_idx}_niid_0_keep_0_test_75.json"
    with open(train_file_path, 'r') as f:
        json_dict = json.load(f)
        users = json_dict["users"]
        user = random.choice(users)
        while user in chosen_user_list:
            user = random.choice(users)
        user_samples = json_dict["num_samples"]
        idx = users.index(user)
        print(user, "train samples:", user_samples[idx])
        train_x = np.array(json_dict['user_data'][user]['x']).reshape((user_samples[idx], 1, 28, 28))
        train_y = np.array(json_dict['user_data'][user]['y'])
        train_data.append({'x': train_x, 'y': train_y})
    with open(test_file_path, 'r') as f:
        json_dict = json.load(f)
        users = json_dict["users"]
        user_samples = json_dict["num_samples"]
        idx = users.index(user)
        print(user, "test samples", user_samples[idx])
        test_x = np.array(json_dict['user_data'][user]['x']).reshape((user_samples[idx], 1, 28, 28))
        test_y = np.array(json_dict['user_data'][user]['y'])
        test_data.append({'x': test_x, 'y': test_y})
# #
# # # print(json_dict.keys())
# # # # dict_keys(['users', 'num_samples', 'user_data'])
# #
for idx, train_dict in enumerate(train_data):
    with open(train_path + str(idx) + '.npz', 'wb') as f:
        np.savez_compressed(f, data=train_dict)
for idx, test_dict in enumerate(test_data):
    with open(test_path + str(idx) + '.npz', 'wb') as f:
        np.savez_compressed(f, data=test_dict)

print("generating global test")
global_test_x = []
global_test_y = []
# randomly choose 4 users as the global test
for i in range(4):
    user_idx = random.randint(0, 35)  # first choose which file to read
    train_file_path = f"/home/zpy/Others/leaf/data/femnist/data/train/all_data_{user_idx}_niid_0_keep_0_train_75.json"
    test_file_path = f"/home/zpy/Others/leaf/data/femnist/data/test/all_data_{user_idx}_niid_0_keep_0_test_75.json"
    with open(train_file_path, 'r') as f:
        json_dict = json.load(f)
        users = json_dict["users"]
        user = random.choice(users)
        while user in chosen_user_list:
            user = random.choice(users)
        user_samples = json_dict["num_samples"]
        idx = users.index(user)
        print(user, "train samples:", user_samples[idx])
        train_x = np.array(json_dict['user_data'][user]['x']).reshape((user_samples[idx], 1, 28, 28))
        train_y = np.array(json_dict['user_data'][user]['y'])
        global_test_x.extend(train_x)
        global_test_y.extend(train_y)
    with open(test_file_path, 'r') as f:
        json_dict = json.load(f)
        users = json_dict["users"]
        user_samples = json_dict["num_samples"]
        idx = users.index(user)
        print(user, "test samples", user_samples[idx])
        test_x = np.array(json_dict['user_data'][user]['x']).reshape((user_samples[idx], 1, 28, 28))
        test_y = np.array(json_dict['user_data'][user]['y'])
        global_test_x.extend(test_x)
        global_test_y.extend(test_y)

test_dict = {'x': global_test_x, 'y': global_test_y}

with open(dir_path + "global_test", 'wb') as f:
    np.savez_compressed(f, data=test_dict)
