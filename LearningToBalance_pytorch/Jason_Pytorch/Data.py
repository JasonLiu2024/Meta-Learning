import numpy as np
from torch.utils.data import dataset, sampler, dataloader
import torch
import matplotlib.pyplot as plt
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUMBER_OF_SAMPLES_EA_CLASS = 20
LABEL_NAMES : list[str] = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']
DATA_FOLDER = "../data/sorted_by_class"

"""reference: the original implementation of MAML"""

def show(image_array, title=""):
    f, ax = plt.subplots(figsize=(2, 2))
    ax.imshow(np.squeeze(image_array))
    ax.set_title(title)

class LearningToBalanaceDataset(dataset.Dataset):
  """a meta learning dataset; items are TASKs"""
  def __init__(self, max_number_of_shot : int, number_of_query : int, 
               way : int, task_dictionary : dict):
    super().__init__()
    # assert number_of_support + number_of_query <= NUMBER_OF_SAMPLES_EA_CLASS, "LearningToBalanaceDataset: support + query > number of samples ea class!"
    self.way = way
    self.max_number_of_shot = max_number_of_shot
    self.number_of_query = number_of_query
    self.dictionary = task_dictionary
  # def __getitem__(self, task_index):
  #   """class_indices: indices of classes we sample examples FROM
  #   there are WAY number of entries in class_indices
  #   there will be SHOT number of samples per class here"""
  #   # NOTE: support-query pair is specific for each stage (train valid test)
  #   # e.g. NOT okay if support set from train section, query set from valid section while in train stage
  #   images_support = self.data_dictionary[task_index]['image support']
  #   labels_support = self.data_dictionary[task_index]['label support']
  #   images_query = self.data_dictionary[task_index]['image query']
  #   labels_query = self.data_dictionary[task_index]['image query']
  #   return images_support, labels_support, images_query, labels_query
  def __getitem__(self, class_indices):
    # print(f"LearningToBalanceDataset: getting item from __getitem__")
    # generate either task or class imbalance, by coin probability
    coin = np.random.uniform(low=0, high=1, size=1)
    # 'shot' is an array -> there's a specific shot for each class!
    if coin > 0.5:
        # print(f"\t->class imbalance: diff shots btwn classes")
        # class imbalanace: different number of shots between classes
        shot = np.random.choice(range(1, self.max_number_of_shot), size=self.way, replace=True)
    else: 
        # print(f"\t->task imbalance: same shots for ea class in this task")
        # task imbalance: different number of shots between tasks
        # BUT same number of shots WITHIN this task
        shot = np.random.choice(range(1, self.max_number_of_shot), size=1)
        shot = shot.repeat(repeats=self.way)
    images_support = []
    labels_support = []
    images_query = []
    labels_query = []
    for label, class_index in enumerate(class_indices):
        # we're going class-by-class here!
        # the enumerating index is the label
        # we get labels like [12, 5, 6, 9, 7]
        # BUT, cross-entropy loss accepts only integer labels in range [0, number of classes]
        # this triggers the error CUDA error: device-side assert triggered
        # therefore, we just use the index, which ensures that each class has unique corresponding label
        # which makes loss calculation work!
        # print(f"working on {i}th class {LABEL_NAMES[class_index]}")
        total_number_of_examples_in_class = len(self.dictionary[class_index]['image'])
        # number of shots to use
        if self.max_number_of_shot + self.number_of_query > total_number_of_examples_in_class:
            raise ValueError("LearningToBalanceDataset: shots + query > total sample count!")
        actual_shot = shot[label]
        total_samples = actual_shot + self.number_of_query
        sample_indices = list(np.random.choice(range(total_number_of_examples_in_class), size=total_samples, replace=False))
        # print(f"we're sampling {actual_shot} + {self.number_of_query} = {total_samples} classes")
        # print(f"adding below to images_support")
        # print(f"{self.dictionary[class_index]['image'][sample_indices[:actual_shot]]}")
        # show(self.dictionary[class_index]['image'][sample_indices[:actual_shot]][0],
        #      LABEL_NAMES[self.dictionary[class_index]['label'][sample_indices[:actual_shot]][0]])
        # break
        """count of support images are varying, due to sample -> need to PAD with all-zeros!
        because dataloader calls torch.stack on each item in the returning tuple across all returning tuples"""
        to_be_padded = self.dictionary[class_index]['image'][sample_indices[:actual_shot]].reshape([-1, 3, 32, 32])
        # print(f"pre-pad image support {to_be_padded.shape}")
        if actual_shot < self.max_number_of_shot:
            pad_amount = self.max_number_of_shot - actual_shot
            # print(f"pad amount: {pad_amount}")
            image_support_padded = np.concatenate([to_be_padded,
                np.zeros(shape=(pad_amount, to_be_padded.shape[1], to_be_padded.shape[2], to_be_padded.shape[3]))])
            label_support_padded = [label] * actual_shot + [0] * pad_amount
        else:
            image_support_padded = to_be_padded
            label_support_padded = [label] * actual_shot
        # print(f"padded image support: {image_support_padded.shape}")
        images_support.extend(torch.tensor(image_support_padded, dtype=torch.float32))
        labels_support.extend(label_support_padded)
        # print(f"labels_support: {labels_support}")
        # labels_support.append(self.dictionary[class_index]['label'][sample_indices[:actual_shot]])
        """count of query images are fixed, no need to pad"""
        images_query.extend(torch.tensor(self.dictionary[class_index]['image'][sample_indices[actual_shot:]].reshape([-1, 3, 32, 32]), dtype=torch.float32))
        labels_query.extend([label] * self.number_of_query)
        # labels_query.append(self.dictionary[class_index]['label'][sample_indices[actual_shot:]])
    # print(f"\tgot {len(class_indices)} classes")
    images_support = torch.stack(images_support)
    labels_support = torch.tensor(labels_support)
    images_query = torch.stack(images_query)
    labels_query = torch.tensor(labels_query)
    return images_support.to(DEVICE), labels_support.to(DEVICE), images_query.to(DEVICE), labels_query.to(DEVICE)

class LearningToBalanceSampler(sampler.Sampler):
    def __init__(self, indices_to_sample_from, way, total_tasks):
        super().__init__(None)
        # print(f"Sampler indices: \n{indices_to_sample_from}")
        self.indices = indices_to_sample_from
        self.way = way
        self.total_tasks = total_tasks
    def __iter__(self):
        sampled = (np.random.default_rng().choice(
                self.indices,
                size=self.way,
                replace=False
            ) for _ in range(self.total_tasks))
        # print(f"sampler: {sampled}")
        return sampled
    def __len__(self):
        return self.total_tasks

def identity(x):
    return x

def get_dataloader(class_names : list[str], max_number_of_shot : int, 
                   number_of_query : int, way : int, total_tasks : int,
                   batch_size : int):
    """total_task: e.g. in training, total_task = number of training tasks"""
    data_dictionary = {}
    l = 0
    # print(f"loading up {len(class_names)} classes, counting")
    class_labels = []
    for class_name in class_names:
      class_data = np.load(f"{DATA_FOLDER}/{class_name}.npy")
      class_label = LABEL_NAMES.index(class_name)
      class_labels.append(class_label)
      l += len(class_data)
    #   print(f"\t+{len(class_data)}")
      data_dictionary[class_label] = {}
      data_dictionary[class_label]['image'] = class_data
      data_dictionary[class_label]['label'] = np.repeat(class_label, repeats=l)
    # print(f"total number of samples: {l}")
    return dataloader.DataLoader(
       dataset=LearningToBalanaceDataset(max_number_of_shot, number_of_query, way, data_dictionary),
       sampler=LearningToBalanceSampler(indices_to_sample_from=class_labels, way=way, total_tasks=total_tasks),
       drop_last=True,
       batch_size=batch_size,
       collate_fn=identity,
    )
