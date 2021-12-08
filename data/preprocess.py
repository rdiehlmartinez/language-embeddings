
''' Processes data in raw multilingual parallel corpus stored under raw folder.

Data used is the TED Multilingual Parallel Corpus: 
https://github.com/ajinkyakulkarni14/TED-Multilingual-Parallel-Corpus

Splits the initial data into files of different languages
'''

import os


def main():
    
    out_data_dir = "preprocessed_data"

    if os.path.isdir(out_data_dir):
        raise Exception(f"{out_data_dir} directory already exists")

    os.mkdir(out_data_dir)

    language_files = {}

    write_ar = True
    prev_line = "" #keeps track of the previous line

    with open("raw_multilingual_parallel/corpus.txt") as f: 
        lines = f.readlines()

        for line_idx, line in enumerate(lines):

            if (line_idx % 10_000 == 0):
                print(f"Processed {line_idx} number of lines")

            line_split = line.strip().split(':')
            try:
                line_lng = line_split[1]
                line_txt = "".join(line_split[2:])

                # called in order to raise ValueError exception 
                line_number = int(line_split[0]) 

            # except statements to catch continuation of current line
            except IndexError:
                # skip if current line is continuation of the previous line
               continue
            except ValueError:
                # the first entry cannot be converted to an int 
                continue 

        
            # Checking out next line to see if continuation on next line
            # NOTE: we only assume that a line continues up to 1 additional line below (not multiple)
            try:
                next_line = lines[line_idx+1]
                next_line_split = next_line.strip().split(':')
                if len(next_line_split) == 1:
                    # the 0th element should just be the continuation of the previous line 
                    line_txt = line_txt + ' ' + next_line_split[0] 
            except: 
                pass

            if (line_lng not in language_files):
                language_files[line_lng] = open(f"{out_data_dir}/{line_lng}.txt", 'a')

            out_file = language_files[line_lng]
            if line_lng == "ar":
                # writing out arabic only every other time 
                if write_ar: 
                    out_file.write(line_txt + '\n')
                    write_ar = False
                else: 
                    write_ar = True
            else: 
                out_file.write(line_txt + '\n')

if __name__ == '__main__':
    main()