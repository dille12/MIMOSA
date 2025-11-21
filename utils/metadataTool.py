import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image
from PIL.TiffImagePlugin import IFDRational
from metadataExtract import flatten_dict
import tifffile
class TiffTagViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("TIFF Tag Viewer")
        self.root.geometry("900x600")
        
        # Buttons
        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=10)
        tk.Button(btn_frame, text="Load TIFF", command=self.load_tiff).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Copy Selected Key", command=self.copy_key).pack(side=tk.LEFT, padx=5)
        
        # Search bar
        search_frame = tk.Frame(root)
        search_frame.pack(pady=5, fill=tk.X, padx=10)
        tk.Label(search_frame, text="Search:").pack(side=tk.LEFT)
        self.search_var = tk.StringVar()
        search_entry = tk.Entry(search_frame, textvariable=self.search_var)
        search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.search_var.trace_add("write", lambda *args: self.filter_tree())
        
        # Treeview 
        tree_frame = tk.Frame(root)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        scrollbar_y = tk.Scrollbar(tree_frame)
        scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        scrollbar_x = tk.Scrollbar(tree_frame, orient=tk.HORIZONTAL)
        scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.tree = ttk.Treeview(tree_frame, columns=("Value",), 
                                 yscrollcommand=scrollbar_y.set,
                                 xscrollcommand=scrollbar_x.set)
        self.tree.pack(fill=tk.BOTH, expand=True)
        scrollbar_y.config(command=self.tree.yview)
        scrollbar_x.config(command=self.tree.xview)
        
        self.tree.heading("#0", text="Tag Name")
        self.tree.heading("Value", text="Value")
        self.tree.column("#0", width=400)
        self.tree.column("Value", width=450)
        self.tree.bind("<Double-Button-1>", lambda e: self.copy_key())
        
        # store flattened dict for filtering
        self.flattened_tags = {}

    def load_tiff(self):
        filepath = filedialog.askopenfilename(
            title="Select TIFF file",
            filetypes=[("TIFF files", "*.tif *.tiff"), ("All files", "*.*")]
        )
        if not filepath:
            return
        
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        try:
            print("LOADING FROM METADATA")
            with tifffile.TiffFile(filepath) as tif:
                page = tif.pages[0]
                tags = {tag.name: tag.value for tag in page.tags.values()}

            self.flattened_tags = flatten_dict(tags)
            self.filter_tree()  # display all initially
            
        except Exception as e:
            self.tree.insert("", tk.END, text="ERROR", values=(str(e),))
    
    def filter_tree(self):
        query = self.search_var.get().lower()
        # clear tree
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        for key in sorted(self.flattened_tags.keys()):
            if query in key.lower():
                value = self.flattened_tags[key]
                
                if isinstance(value, IFDRational):
                    display_value = f"{float(value):.6f} ({value.numerator}/{value.denominator})"
                elif isinstance(value, bytes):
                    if len(value) > 50:
                        display_value = f"[{len(value)} bytes] {value[:50]}..."
                    else:
                        display_value = str(value)
                else:
                    display_value = str(value)
                
                self.tree.insert("", tk.END, text=key, values=(display_value,))

    def copy_key(self):
        selection = self.tree.selection()
        if selection:
            item = selection[0]
            key = self.tree.item(item, "text")
            self.root.clipboard_clear()
            self.root.clipboard_append(key)


if __name__ == "__main__":
    root = tk.Tk()
    app = TiffTagViewer(root)
    root.mainloop()