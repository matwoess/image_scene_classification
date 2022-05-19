from pathlib import Path
from PIL import Image

if __name__ == '__main__':
    # p = Path('data/seg_train')
    # images = list(p.rglob('*.jpg'))
    # for img in images:
    #     image = Image.open(img)
    #     if image.size != (150, 150):
    #         print(f'{image.size} - {img}')
    d1 = {'a': 1, 'b': 2}
    d2 = {'c': 3}
    d1.update(d2)
    print(d1)