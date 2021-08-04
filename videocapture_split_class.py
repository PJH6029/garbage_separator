import cv2
import glob

invideo_path = glob.glob('data/video/*') #->can, pet
save_path = glob.glob('saving/trash_split_class/*') #->can, pet

def video2frame(invideo_path, save_path):
    local_count = 0
    #invdeo_path, save_path: 폴더 주소

    invideo_directory = glob.glob(invideo_path+'/*.mp4') #-> 폴더 안 영상

    for video in invideo_directory: #video: mp4파일

        vidcap = cv2.VideoCapture(video)

        while True:
            sucess, img = vidcap.read()
            if not sucess:
                break
            if int(vidcap.get(1) % 20) == 0:
                print('Read a new frame:', sucess, local_count)
                frame = '/{0}{1}.jpg'.format('0', '{0:05d}'.format(local_count))
                cv2.imwrite(save_path+frame, img)
                local_count +=1

        print('next video')

    print('sucess')

def main():
    print(save_path)
    #for num, save_directory in enumerate(save_path): #->save directory: lemon, suckru...
     #   video2frame(invideo_path, save_directory, num)
    for i in range(len(save_path)):
        #video2frame(invideo_path[i], save_path[i], i) #->aloe
        invideo = glob.glob(invideo_path[i]+'/*') #->invideo에 aloe...
        save = glob.glob(save_path[i] + '/*')
        print(invideo, save)
        for num in range(len(invideo)):
            video2frame(invideo[num], save[num]) #->invideo[num]: aloe디렉토리 주소
            print(invideo[num], save[num])


if __name__ == '__main__':
    main()