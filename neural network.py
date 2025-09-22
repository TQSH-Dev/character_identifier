import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
import os
from PIL import Image, ImageTk

GRID_DIM = 28
PIXEL_SIZE = 20
WINDOW_SIZE = GRID_DIM * PIXEL_SIZE

BRUSH_RADIUS = 2 
BRUSH_INTENSITY = 0.5 

class GrayscaleApp:
    def __init__(self, root):
        self.root = root
        self.root.title("28x28 Grayscale Grid")
        self.activity_grid = np.zeros((GRID_DIM, GRID_DIM), dtype=np.float64)

        self.canvas = tk.Canvas(root, width=WINDOW_SIZE, height=WINDOW_SIZE, bg='black')
        self.canvas.pack()

        button_frame = tk.Frame(root)
        button_frame.pack(pady=10)

        self.activity_button = tk.Button(button_frame, text="Get Activity", command=self.get_activity_grid)
        self.activity_button.pack(side=tk.LEFT, padx=5)

        self.clear_button = tk.Button(button_frame, text="Clear", command=self.clear_grid)
        self.clear_button.pack(side=tk.LEFT, padx=5)

        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<Button-1>", self.draw)

        self.photo_image = None
        self.update_canvas()

    def draw(self, event):
        col = event.x // PIXEL_SIZE
        row = event.y // PIXEL_SIZE

        y, x = np.ogrid[-BRUSH_RADIUS:BRUSH_RADIUS + 1, -BRUSH_RADIUS:BRUSH_RADIUS + 1]
        mask = x**2 + y**2 <= BRUSH_RADIUS**2
        distance_sq = x**2 + y**2
        sigma_sq = (BRUSH_RADIUS / 2)**2 
        gaussian = BRUSH_INTENSITY * np.exp(-distance_sq / (2 * sigma_sq)) * mask

        min_row = max(0, row - BRUSH_RADIUS)
        max_row = min(GRID_DIM, row + BRUSH_RADIUS + 1)
        min_col = max(0, col - BRUSH_RADIUS)
        max_col = min(GRID_DIM, col + BRUSH_RADIUS + 1)

        brush_min_row = min_row - (row - BRUSH_RADIUS)
        brush_max_row = max_row - (row - BRUSH_RADIUS)
        brush_min_col = min_col - (col - BRUSH_RADIUS)
        brush_max_col = max_col - (col - BRUSH_RADIUS)

        self.activity_grid[min_row:max_row, min_col:max_col] += gaussian[brush_min_row:brush_max_row, brush_min_col:brush_max_col]

        np.clip(self.activity_grid, 0.0, 1.0, out=self.activity_grid)
        
        self.update_canvas()

    def update_canvas(self):
        img_data = (self.activity_grid * 255).astype(np.uint8)

        pil_image = Image.fromarray(img_data, mode='L')

        pil_image = pil_image.resize((WINDOW_SIZE, WINDOW_SIZE), Image.NEAREST)
        
        self.photo_image = ImageTk.PhotoImage(image=pil_image)

        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo_image)

    def get_activity_grid(self):
        return self.activity_grid

    def clear_grid(self):
        self.activity_grid.fill(0.0)
        self.update_canvas()

if __name__ == "__main__":
    main_window = tk.Tk()
    app = GrayscaleApp(main_window)
    main_window.mainloop()
    initial_activities=app.get_activity_grid()

def sigmoid(x):
    return 1/(1+np.exp(-x))

w=[]
neurallayers=[]
numoflays=4

class neuron:
    def __init__(self,prevlayer,nextlayer,prevn,countn,currentn):
        self.prevlayer=prevlayer
        self.nextlayer=nextlayer
        self.prevn=prevn
        self.countn=countn
        self.currentn=currentn
        self.activity=1
        if prevn!=0:
            self.weights=np.array([1 for i in range(self.prevn)])
            self.bias=1
        else:
            self.activity=initial_activities[currentn//28][currentn%28]
            self.weights=None
            self.bias=None
    def updateweightsbiases(cls):
        w.append(cls.weights)
    def calcactivity(cls):
        
        if cls.prevlayer!=None:
            prevactivities=np.array([i.activity for i in neurallayers[cls.prevlayer].neurons])
            cls.activity= sigmoid (np.sum(cls.weights*prevactivities)+cls.bias)

class neurallayer:
    def __init__(self,prevlayer,nextlayer,prevn,countn):
        self.prevlayer=prevlayer
        self.nextlayer=nextlayer
        self.prevn=prevn
        self.countn=countn
        self.neurons=[]
        for i in range(countn):
            self.neurons.append(neuron(self.prevlayer,self.nextlayer,self.prevn,self.countn,i)) 
            neuron.calcactivity(self.neurons[i])
            if prevlayer!=None:
                neuron.updateweightsbiases(self.neurons[i])
                
       
neurallayers.append(neurallayer(None,1,0,784))
for i in range(1,numoflays-1):
    neurallayers.append(neurallayer(i-1,i+1,neurallayers[i-1].countn,16))
neurallayers.append(neurallayer(numoflays-2,None,neurallayers[numoflays-2].countn,10))


file='weights_biases.npz'
folder='data'
os.makedirs(folder, exist_ok=True) 

path=os.path.join(folder,file)
np.savez(path,*w)
data=np.load(path)
for i in data:
    print(data[i])
print(path)





        
        

