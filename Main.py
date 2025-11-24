import Processor
import Gui

if __name__ == '__main__':
    #GUI = Gui.GUI()
    #GUI.run()

    for i in range (1,10):
        img_path = "./data/" + str(i) + ".jpg"
        #Processor.run(img_path)
        Processor.execute(img_path)


    ''' 

    disp = my_img.copy()

    blue = np.zeros_like(disp)
    blue[:, :, 0] = 255
    disp = np.where(mold_mask[..., None] > 0, cv2.addWeighted(disp, 0.6, blue, 0.4, 0), disp)

    plt.figure(figsize=(10, 6))
    plt.axis('off')
    plt.imshow(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB))
    plt.title('Combined')
    plt.show()
    '''