# coding: utf-8
import os
def main():
    wrk_dir = r"D:\hcho_change\batch_codes\in"
    os.chdir(wrk_dir)
    files = os.listdir(os.getcwd())
    for file in files:
        if (" " in file) == True:
            print file
            os.rename(file, file.split(" ")[0] + "0" + file.split(" ")[1])

if __name__ == '__main__':
    main()
    print "ok"