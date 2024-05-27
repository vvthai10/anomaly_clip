import os
import json

import os
import json


class MpddSolver(object):
    CLSNAMES = ['chest']

    def __init__(self, root='data/mvtec'):
        self.root = root
        self.meta_path = f'{root}/meta.json'

    def run(self):
        info = dict(train={}, test={})
        anomaly_samples = 0
        normal_samples = 0
        for cls_name in self.CLSNAMES:
            cls_dir = f'{self.root}'
            for phase in ['test']:
                cls_info = []
                species = os.listdir(f'{cls_dir}')
                for specie in species:
                  if ".ipynb" in specie:
                    continue
                  is_abnormal = True if specie not in ['NORMAL', 'Normal'] else False
                  img_names = os.listdir(f'{cls_dir}/{specie}/images/')

                  img_names.sort()

                  for idx, img_name in enumerate(img_names):
                      img_path = f'{specie}/images/{img_name}'
                      info_img = dict(
                          img_path=img_path,
                          mask_path=img_path.replace("/images/","/masks/"),
                          cls_name=cls_name,
                          specie_name=specie,
                          anomaly=1 if is_abnormal else 0,
                      )
                      cls_info.append(info_img)
                      if phase == 'test':
                          if is_abnormal:
                              anomaly_samples = anomaly_samples + 1
                          else:
                              normal_samples = normal_samples + 1
                info[phase][cls_name] = cls_info
        with open(self.meta_path, 'w') as f:
            f.write(json.dumps(info, indent=4) + "\n")
        print('normal_samples', normal_samples, 'anomaly_samples', anomaly_samples)

if __name__ == '__main__':
    runner = MpddSolver(root='./data/medical_test/COVID-19')
    runner.run()
