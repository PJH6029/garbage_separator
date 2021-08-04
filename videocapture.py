import cv2
import glob

invideo_path = glob.glob('data/video/*')
save_path = glob.glob('saving/trash/*')

def video2frame(invideo_path, save_path, total_count):
    local_count = 0

    invideo_directory = glob.glob(invideo_path[total_count]+'/*.mp4')

    for video in invideo_directory:

        vidcap = cv2.VideoCapture(video)

        while True:
            sucess, img = vidcap.read()
            if not sucess:
                break
            if int(vidcap.get(1) % 20) == 0:
                print('Read a new frame:', sucess, local_count)
                frame = '/{0}{1}.jpg'.format(total_count, '{0:05d}'.format(local_count))
                cv2.imwrite(save_path+frame, img)
                local_count +=1

        print('next video')

    print('sucess')

def main():
    print(save_path)
    for num, save_directory in enumerate(save_path):
        video2frame(invideo_path, save_directory, num)


if __name__ == '__main__':
    main()