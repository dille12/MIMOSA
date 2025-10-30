import clipboard
i = clipboard.paste()
i = i.replace("\\", "/")
print(i)
clipboard.copy(i)