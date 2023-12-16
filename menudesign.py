from tkinter import *
root=Tk()
root.geometry('300x300')
l=Label(root,text="Smart Camera")
l.pack()
l1=Label(root,text="Choose an option")

def but():
    exec(open("./open.py").read())
b=Button(root,text="Face Recognition",command=but)
b.pack()

root.mainloop()