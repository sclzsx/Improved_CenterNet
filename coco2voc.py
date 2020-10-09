from pycocotools.coco import COCO
import skimage.io as io
import matplotlib.pyplot as plt
import pylab, os, cv2, shutil
from lxml import etree, objectify
from tqdm import tqdm
import random
from PIL import Image

pylab.rcParams['figure.figsize'] = (8.0, 10.0)

coco_dir = '/workspace/DATA/zhatu/zhatu0814/coco'
adopted_cats = ['soil']

voc_dir = '/workspace/DATA/zhatu/zhatu0814/voc'
voc_img_dir = voc_dir + "/" + "JPEGImages"
voc_ann_dir = voc_dir + "/" + "Annotations"


def mkr(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def showimg(coco, dataType, img, CK5Ids):
    global coco_dir
    I = io.imread('%s/%s/%s' % (coco_dir, dataType, img['file_name']))
    plt.imshow(I)
    plt.axis('off')
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=CK5Ids, iscrowd=None)
    anns = coco.loadAnns(annIds)
    coco.showAnns(anns)
    plt.show()


def save_annotations(dataType, filename, objs):
    annopath = voc_ann_dir + "/" + filename[:-3] + "xml"
    img_path = coco_dir + "/image/" + filename
    dst_path = voc_img_dir + "/" + filename
    img = cv2.imread(img_path)
    im = Image.open(img_path)
    if im.mode != "RGB":
        print(filename + " not a RGB image")
        im.close()
        return
    im.close()
    shutil.copy(img_path, dst_path)
    E = objectify.ElementMaker(annotate=False)
    anno_tree = E.annotation(
        E.folder('1'),
        E.filename(filename),
        E.source(
            E.database('CKdemo'),
            E.annotation('VOC'),
            E.image('CK')
        ),
        E.size(
            E.width(img.shape[1]),
            E.height(img.shape[0]),
            E.depth(img.shape[2])
        ),
        E.segmented(0)
    )
    for obj in objs:
        E2 = objectify.ElementMaker(annotate=False)
        anno_tree2 = E2.object(
            E.name(obj[0]),
            E.pose(),
            E.truncated("0"),
            E.difficult(0),
            E.bndbox(
                E.xmin(obj[2]),
                E.ymin(obj[3]),
                E.xmax(obj[4]),
                E.ymax(obj[5])
            )
        )
        anno_tree.append(anno_tree2)
    etree.ElementTree(anno_tree).write(annopath, pretty_print=True)


def showbycv(coco, dataType, img, classes, CK5Ids):
    global coco_dir
    filename = img['file_name']
    filepath = '%s/image/%s' % (coco_dir, filename)
    I = cv2.imread(filepath)
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=CK5Ids, iscrowd=None)
    anns = coco.loadAnns(annIds)
    objs = []
    for ann in anns:
        name = classes[ann['category_id']]
        if name in adopted_cats:
            if 'bbox' in ann:
                bbox = ann['bbox']
                xmin = (int)(bbox[0])
                ymin = (int)(bbox[1])
                xmax = (int)(bbox[2] + bbox[0])
                ymax = (int)(bbox[3] + bbox[1])
                obj = [name, 1.0, xmin, ymin, xmax, ymax]
                objs.append(obj)
                cv2.rectangle(I, (xmin, ymin), (xmax, ymax), (255, 0, 0))
                cv2.putText(I, name, (xmin, ymin), 3, 1, (0, 0, 255))
    save_annotations(dataType, filename, objs)
    # cv2.imshow("img", I)
    # cv2.waitKey(1)


def catid2name(coco):
    classes = dict()
    for cat in coco.dataset['categories']:
        classes[cat['id']] = cat['name']
        # print(str(cat['id'])+":"+cat['name'])
    return classes


def create_voc():
    mkr(voc_img_dir)
    mkr(voc_ann_dir)
    dataTypes = ['train', 'val']
    for dataType in dataTypes:
        annFile = '{}/{}.json'.format(coco_dir, dataType)
        coco = COCO(annFile)
        adopted_cat_ids = coco.getCatIds(catNms=adopted_cats)
        classes = catid2name(coco)
        for srccat in adopted_cats:
            print(dataType + ":" + srccat)
            catIds = coco.getCatIds(catNms=[srccat])
            imgIds = coco.getImgIds(catIds=catIds)
            # imgIds=imgIds[0:100]
            for imgId in tqdm(imgIds):
                img = coco.loadImgs(imgId)[0]
                showbycv(coco, dataType, img, classes, adopted_cat_ids)
                # showimg(coco,dataType,img,CK5Ids)


# split train and test for training
def split_traintest(trainratio=0.7, valratio=0.2, testratio=0.1):
    dataset_dir = voc_dir
    files = os.listdir(voc_img_dir)
    trains = []
    vals = []
    trainvals = []
    tests = []
    random.shuffle(files)
    for i in range(len(files)):
        filepath = voc_img_dir + "/" + files[i][:-3] + "jpg"
        if (i < trainratio * len(files)):
            trains.append(files[i])
            trainvals.append(files[i])
        elif i < (trainratio + valratio) * len(files):
            vals.append(files[i])
            trainvals.append(files[i])
        else:
            tests.append(files[i])
    # write txt files for yolo
    with open(dataset_dir + "/trainval.txt", "w")as f:
        for line in trainvals:
            line = voc_img_dir + "/" + line
            f.write(line + "\n")
    with open(dataset_dir + "/test.txt", "w") as f:
        for line in tests:
            line = voc_img_dir + "/" + line
            f.write(line + "\n")
    # write files for voc
    maindir = dataset_dir + "/" + "ImageSets/Main"
    mkr(maindir)
    with open(maindir + "/train.txt", "w") as f:
        for line in trains:
            line = line[:line.rfind(".")]
            f.write(line + "\n")
    with open(maindir + "/val.txt", "w") as f:
        for line in vals:
            line = line[:line.rfind(".")]
            f.write(line + "\n")
    with open(maindir + "/trainval.txt", "w") as f:
        for line in trainvals:
            line = line[:line.rfind(".")]
            f.write(line + "\n")
    with open(maindir + "/test.txt", "w") as f:
        for line in tests:
            line = line[:line.rfind(".")]
            f.write(line + "\n")
    print("spliting done")


if __name__ == "__main__":
    create_voc()
    split_traintest()
