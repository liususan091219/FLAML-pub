import os
import shutil

if __name__ == "__main__":
    src_path = "wo_filetag/"
    dst_path = "with_filetag/"
    try:
        shutil.rmtree(dst_path)
        os.mkdir(dst_path)
    except:
        pass
    for each_file in os.listdir(src_path):
        if each_file.startswith("amlk8s"):
            src_file = os.path.join(src_path, each_file)
            dst_file = os.path.join(dst_path, each_file)
            with open(dst_file, "w") as fout:
                with open(src_file, "r") as fin:
                    for line in fin:
                        if line.lstrip(" -").startswith("python"):
                            altered_line = line.rstrip("\n ") + " --yml_file " + each_file + "\n"
                        else:
                            altered_line = line
                        fout.write(altered_line)