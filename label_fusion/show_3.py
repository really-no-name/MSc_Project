import argparse
import os
from PIL import Image
import matplotlib.pyplot as plt


def main(name, data, source, printstate):
    base_path = '/Users/Google_Drive/dataset/' + data + '/output/'
    tissue_path = base_path + 'img_' + name + '.png'
    maps_path = [base_path + f'Maps/Maps{i}_T/pred_{name}_mean.png' for i in range(1, 7)]

    label_fusion_base_path = '/Users/Google_Drive/dataset/' + data + '/label_fusion/' + source
    staple_path = label_fusion_base_path + name + '_STAPLEcolor.png'
    mv_path = label_fusion_base_path + name + '_MV.png'
    av_path = label_fusion_base_path + name + '_AV.png'

    remaining_paths = [
        label_fusion_base_path + name + '_STAPLE_UE.png',
        label_fusion_base_path + name + '_MV_UE.png',
        label_fusion_base_path + name + '_AV_UE.png'
    ]

    output_path = label_fusion_base_path + 'show/' + name + '_show.png'

    # Load images
    tissue_img = Image.open(tissue_path)
    maps_imgs = [Image.open(path) for path in maps_path]
    staple_img = Image.open(staple_path)
    mv_img = Image.open(mv_path)
    av_img = Image.open(av_path)
    remaining_imgs = [Image.open(path).convert('RGB') for path in remaining_paths]  # Convert to RGB

    # Create figure
    fig, axes = plt.subplots(4, max(len(maps_imgs), len(remaining_imgs)), figsize=(16, 16))

    # Adjust layout to reduce spacing
    plt.subplots_adjust(wspace=0.1, hspace=0.1, left=0.05, right=0.95, top=0.95, bottom=0.05)

    # First row: tissue image
    axes[0, 0].imshow(tissue_img)
    axes[0, 0].axis('off')
    for i in range(1, len(axes[0])):
        axes[0, i].axis('off')

    # Second row: maps images
    for i, img in enumerate(maps_imgs):
        axes[1, i].imshow(img)
        axes[1, i].axis('off')
    for i in range(len(maps_imgs), len(axes[1])):
        axes[1, i].axis('off')

    # Third row: staple, mv, av images
    axes[2, 0].imshow(staple_img)
    axes[2, 0].axis('off')
    axes[2, 1].imshow(mv_img)
    axes[2, 1].axis('off')
    axes[2, 2].imshow(av_img)
    axes[2, 2].axis('off')
    for i in range(3, len(axes[2])):
        axes[2, i].axis('off')

    # Fourth row: remaining images
    for i, img in enumerate(remaining_imgs):
        axes[3, i].imshow(img)
        axes[3, i].axis('off')
    for i in range(len(remaining_imgs), len(axes[3])):
        axes[3, i].axis('off')

    # Save the result
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    if printstate == 'on':
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='展示图像')
    parser.add_argument('-ds', '--dataset', type=str, required=True, choices=['Gleason19', 'Arvaniti_TMA'],
                        help='dataset')
    parser.add_argument('-m', '--mode', type=str, required=True, choices=['train', 'test'],
                        help='The mode to run the staple')
    parser.add_argument('-n', '--name', type=str, required=True, help='name')
    parser.add_argument('-p', '--printstate', type=str, default='on', choices=['on', 'off'], help='是否打印及图像显示')
    args = parser.parse_args()

    name = args.name
    data = args.dataset

    if args.mode == 'train':
        source = 'Train_imgs/'
    elif args.mode == 'test':
        source = 'Test_imgs/'

    main(name, data, source, args.printstate)
