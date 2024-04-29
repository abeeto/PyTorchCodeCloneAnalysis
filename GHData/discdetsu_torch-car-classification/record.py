import tkinter as tk
from tkinter import ttk

import pandas as pd

def create_record_window(mainwin):
    
    df = pd.read_csv('record.csv')

    def SaveFile():
        df.to_csv('output.csv',index=False)

    record_win = tk.Toplevel(mainwin)
    record_win.geometry('820x290')
    record_win.title("Record")
    
    # define columns
    columns = ('lpr', 'car_model', 'color', 'entry_date')

    tree = ttk.Treeview(record_win, columns=columns, show='headings')

    # define headings
    tree.heading('lpr', text='License Number')
    tree.heading('car_model', text='Car Model')
    tree.heading('color', text='Color')
    tree.heading('entry_date', text='Entry Date')

    # generate sample data
    del df['Path']
    data = df.values.tolist()

    # add data to the treeview
    for d in data:
        tree.insert('', tk.END, values=d)

    tree.grid(row=0, column=0, sticky='nsew')

    # add a scrollbar
    scrollbar = ttk.Scrollbar(record_win, orient=tk.VERTICAL, command=tree.yview)
    tree.configure(yscroll=scrollbar.set)
    scrollbar.grid(row=0, column=1, sticky='ns')

    save_btn = tk.Button(record_win, text = 'Click to save file ', command = SaveFile)
    save_btn.grid(row=1, column=0, pady = 20,padx = 50)