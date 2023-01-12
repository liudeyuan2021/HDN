import shutil
from pathlib import Path


def delPath(root):
    root = Path(root)
    if root.is_file():
        try:
            root.unlink()
        except Exception as e:
            print(e)
    elif root.is_dir():
        for item in root.iterdir():
            delPath(item)
        try:
            root.rmdir()
            print("Files in {} is removed".format(root))
        except Exception as e:
            print(e)


def mkPath(path):
    p = Path(path)
    try:
        p.mkdir(parents=True, exist_ok=False)
        return True
    except Exception as e:
        return False


def getAllFiles(root, ext=None):
    p = Path(root)
    if ext is not None:
        pathnames = p.glob("**/*.{}".format(ext))
    else:
        pathnames = p.glob("**/*")
    filenames = sorted([x.as_posix() for x in pathnames])
    return filenames


def copyFile(src, dst):
    parent = Path(dst).parent
    mkPath(parent)
    try:
        shutil.copytree(str(src), str(dst))
    except:
        shutil.copy(str(src), str(dst))


def movFile(src, dst):
    parent = Path(dst).parent
    mkPath(parent)
    shutil.move(str(src), str(dst))


def getSubDirs(root):
    p = Path(root)
    dirs = [x for x in p.iterdir() if x.is_dir()]
    return sorted(dirs)


if __name__ == "__main__":
    pass
