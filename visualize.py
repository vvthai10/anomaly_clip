import os
import numpy as np
import matplotlib.pyplot as plt

dataset_path = "./data/train/"
CLASSES = ['brain', 'liver', 'retina_resc']
num_good_samples = []
num_un_good_samples = []

def run():
    for cls in CLASSES:
        cls_dir = f"{dataset_path}/{cls}/"
        for phase in ['test']:
            species = os.listdir(f"{cls_dir}/{phase}")
            for specie in species:
                img_names = os.listdir(f'{cls_dir}/{phase}/{specie}/images')
                if specie == "good":
                    num_good_samples.append(len(img_names))
                else:
                    num_un_good_samples.append(len(img_names))

def visualize():
    x = np.arange(len(CLASSES))
    width = 0.35

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, num_good_samples, width, label='Good')
    rects2 = ax.bar(x + width / 2, num_un_good_samples, width, label='Nogood')

    ax.set_xlabel('Thư mục')
    ax.set_ylabel('Số lượng')
    ax.set_title('Phân bổ dữ liệu trong các thư mục')
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES)
    ax.legend()

    plt.show()

if __name__ == "__main__":
    run()
    print(CLASSES)
    print(num_good_samples)
    print(num_un_good_samples)
    visualize()
