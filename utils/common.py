import os
import config

def get_srcs_tar_name_using_list(all_sub_list_name, subs_path, target_subject, n_class, oracle = False):

    srcs_file_name = 'lab_srcs' + str(len(all_sub_list_name)) + '_cl' + str(n_class) if n_class > 2 else 'lab_srcs' + str(len(all_sub_list_name))
    srcs_file_name = srcs_file_name + "_McMaster" if 'McMaster' in subs_path else srcs_file_name
    rand_list_count = all_sub_list_name[:10] if len(all_sub_list_name) > 10 else all_sub_list_name
    for subject in rand_list_count:
        # sub_folder = all_subs_list[index]
        if 'McMaster' in subs_path:
            split_folder = subject.split("-")
            srcs_file_name = srcs_file_name + "_" + split_folder[0] if subject != target_subject else srcs_file_name
        else:
            split_folder = subject.split("_")
            srcs_file_name = srcs_file_name + "_" + split_folder[1] + split_folder[2] if subject != target_subject else srcs_file_name

    # split_folder = all_subs_list[tar_sub].split("_")
    split_folder = target_subject.split("-") if 'McMaster' in subs_path else target_subject.split("_")

    if 'McMaster' in subs_path:
        tar_file_name = srcs_file_name + "_tar_" + split_folder[0] + "_oracle.txt" if oracle else srcs_file_name + "_tar_" + split_folder[0] + ".txt"
    else:
        tar_file_name = srcs_file_name + "_tar_" + split_folder[0] + split_folder[1] + split_folder[2] + "_oracle.txt" if oracle else srcs_file_name + "_tar_" + split_folder[0] + split_folder[1] + split_folder[2] + ".txt"

    # to create file 
    if len(all_sub_list_name) > 10:
        srcs_file_name = srcs_file_name + "_____only.txt"
    else:
        srcs_file_name = srcs_file_name + "_only.txt"

    return srcs_file_name, tar_file_name, all_sub_list_name

def write_srcs_tar_txt_files_using_list(subs_path, label_file_path, all_sub_list_name, target_subject, n_class, oracle):
  
    srcs_file_name, tar_file_name, _ = get_srcs_tar_name_using_list(all_sub_list_name, subs_path, target_subject, n_class, oracle)

    file_read = open(label_file_path, 'r')
    lines = file_read.readlines()
    
    srcs_write_file = open(srcs_file_name, "w+")
    tar_write_file = open(tar_file_name, "w+")
    for line in lines:
        # read subject folder from LABEL TEXT file and write into a seperate text file
        for sub_folder in all_sub_list_name:
            if sub_folder in line:
                # if 'McMaster' in label_file_path:
                #     mcmaster_data = line.split(" ")
                #     if mcmaster_data[1] == '0' or mcmaster_data[1] == '4' or mcmaster_data[1] == '5':
                #         write_txt = '1' if mcmaster_data[1] == '4' or mcmaster_data[1] == '5' else mcmaster_data[1]
                #         srcs_write_file.write(mcmaster_data[0] + " " + write_txt + "\n") if sub_folder != target_subject else tar_write_file.write(mcmaster_data[0] + " " + write_txt + "\n")
                # else:
                if sub_folder != target_subject:
                    line = line.strip()
                    srcs_write_file.write(line)

                    bio_signal_path = get_bio_path_fr_img(line)
                    srcs_write_file.write(" " + bio_signal_path + "\n")
                else:
                    line = line.strip()
                    tar_write_file.write(line)

                    bio_signal_path = get_bio_path_fr_img(line)
                    tar_write_file.write(" " + bio_signal_path + "\n")

    srcs_write_file.close()
    tar_write_file.close()

    return all_sub_list_name

def load_biovid_src_subs(topk=None):
    subject_list = None
    if topk is None:
        subject_list = ['082208_w_45', '081714_m_36', '112610_w_60', '101908_m_61', '071709_w_23','082014_w_24', '110810_m_62', '080209_w_26', 
                        '101916_m_40', '110614_m_42', '101814_m_58', '112016_m_25', '071313_m_41', '102514_w_40', '100514_w_51', '101114_w_37', 
                        '100509_w_43', '082315_w_60', '112310_m_20', '120614_w_61', '092714_m_64', '101514_w_36', '092813_w_24', '102414_w_58', 
                        '102309_m_61', '081617_m_27', '080609_w_27', '083114_w_55', '111313_m_64', '071614_m_20', '101309_m_48', '071911_w_24', 
                        '102316_w_50', '100417_m_44', '083013_w_47', '083009_w_42', '080714_m_23', '101809_m_59', '082909_m_47', '101209_w_61', 
                        '092014_m_56', '072414_m_23', '101015_w_43', '112909_w_20', '111609_m_65', '100117_w_36', '111409_w_63', '080709_m_24', 
                        '072714_m_23', '112914_w_51', '120514_w_56', '083109_m_60', '110909_m_29', '091814_m_37', '071814_w_23', '092509_w_51', 
                        '112809_w_23', '100214_m_50', '102214_w_36', '082714_m_22', '082109_m_53', '092808_m_51', '080309_m_29', '102008_w_22', 
                        '111914_w_63', '082809_m_26', '072514_m_27', '082814_w_46', '072609_w_23', '101216_m_40', '091914_m_46', '100914_m_39', 
                        '112209_m_51', '092514_m_50', '092009_m_54', '082414_m_64', '080614_m_24']
    return subject_list


def get_bio_path_fr_img(img_path:str):
    infos = img_path.split("/")
    bio_path = config.BIO_DATASET + "/"+ infos[1] + "/" + infos[2] + "_bio.csv"
    return bio_path