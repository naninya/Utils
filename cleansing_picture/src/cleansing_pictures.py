import os
import cv2
import argparse


def cleansing_picture(input_dir, resize_width):
    imgs = {}
    for file_name in os.listdir(input_dir):
        # print(file_name)
        input_path = os.path.join(input_dir, file_name)
        img = cv2.imread(input_path)
        height = img.shape[0]
        width = img.shape[1]

        hw_rate = height / width
        # resize
        if resize_width > img.shape[1]:
            resize_width = img.shape[1]
        else:
            # print("resized!")
            img = cv2.resize(img, (resize_width, int(hw_rate * resize_width)), interpolation=cv2.INTER_AREA)

        # crop white space
        for index, coor_pos in enumerate(["height", "width"]):
            valid_line_index = []
            if coor_pos == "height":
                for col in range(img.shape[index]):
                    if not all((img[col, :, :] == 255).flatten()):
                        valid_line_index.append(col)

                img = img[valid_line_index, :, :]
            else:
                for row in range(img.shape[index]):
                    if not all((img[:, row, :] == 255).flatten()):
                        valid_line_index.append(row)
                img = img[:, valid_line_index, :]
        
        imgs[file_name] = img
    return imgs

    
def save_cleansing_picture():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="./input")
    parser.add_argument("--resize_width", type=str, default=150)
    args = parser.parse_args()
    input_dir = args.input_dir
    
    resize_width = args.resize_width
    if not os.path.isdir("../output"):
        os.mkdir("../output")

    for file_name, img in cleansing_picture(input_dir, resize_width).items():
        output_path = f"../output/resized_{file_name}"
        cv2.imwrite(output_path, img)


if __name__ == "__main__":
    save_cleansing_picture()