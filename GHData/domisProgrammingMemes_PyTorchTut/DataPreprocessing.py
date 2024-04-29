import pandas as pd
import openpyxl as pyxl

# allpeak = pyxl.load_workbook("./data/FaceTracker/Excel/Tim_2410_AllPeak.xlsx")
# sheets = ["Neutral_Face", "Neutral_Face_centered", "AU1L", "AU1R", "AU2L", "AU2R", "AU4L", "AU4R"]
# allsheets = allpeak.get_sheet_names()
# deletesheets = list(set(allsheets) - set(sheets))
# print(deletesheets)
# for ind, data in enumerate(deletesheets):
#     del allpeak[data]
# allpeak.save("data/FaceTracker/Processed/Excel/AU1AU2AU4.csv")

au1au2 = pd.read_csv("data/FaceTracker/CSV/Tim_2410_AU1AU2.csv")
au1au2.sort_values(by='Frame', ascending=True, inplace=True)
au1au2.drop_duplicates(subset='Frame', inplace=True, ignore_index=True)

au4 = pd.read_csv("data/FaceTracker/CSV/Tim_2410_AU4.csv")
au4.sort_values(by='Frame', ascending=True, inplace=True)
au4.drop_duplicates(subset='Frame', inplace=True, ignore_index=True)

au9 = pd.read_csv("data/FaceTracker/CSV/Tim_2410_AU9.csv")
au9.sort_values(by='Frame', ascending=True, inplace=True)
au9.drop_duplicates(subset='Frame', inplace=True, ignore_index=True)

rows = []
missingframes = []

# check if every frame is included
x = 1
for idx, data in au9.iterrows():
    # works but not the right way of doing it
    # rows.append(au1au2["Frame"].iloc[idx])
    # better:
    # rows.append(data[0])

    # which frames are missing?
    if float(idx) != (data[0] - x):
        missingframes.append(idx+x)
        # print(f"frame {idx + x} does not exist")
        x += 1


print()
print(missingframes)

count1 = 0
count2 = 0
count3 = 0
count4 = 0
count5 = 0
count6 = 0


for i in range(len(missingframes)):
    try:
        if missingframes[i] + 1 == (missingframes[i+1]):
            print(f"1 frame skipped {missingframes[i]}")
            count1 += 1
        elif missingframes[i] + 2 == (missingframes[i + 1]):
            print(f"2 frames skipped {missingframes[i]}")
            count2 += 1
        elif missingframes[i] + 3 == (missingframes[i+1]):
            print(f"3 frames skipped {missingframes[i]}")
            count3 += 1
        elif missingframes[i] + 4 == (missingframes[i + 1]):
            print(f"4 frames skipped {missingframes[i]}")
            count4 += 1
        elif missingframes[i] + 5 == (missingframes[i + 1]):
            print(f"5 frames skipped {missingframes[i]}")
            count5 += 1
        elif missingframes[i] + 6 == (missingframes[i + 1]):
            print(f"6 frames skipped {missingframes[i]}")
            count6 += 1
        else:
            print("more than 6 frames skipped")
    except IndexError:
        print("end of file")

print(f"1 frame skipped {count1} times")
print(f"2 frame skipped {count2} times")
print(f"3 frame skipped {count3} times")
print(f"4 frame skipped {count4} times")
print(f"5 frame skipped {count5} times")
print(f"6 frame skipped {count6} times")

# safe
# au1au2.to_csv(path_or_buf="data/FaceTracker/Processed/CSV/au1au2.csv", index=False)
# au4.to_csv(path_or_buf="data/FaceTracker/Processed/CSV/au4.csv", index=False)
# au9.to_csv(path_or_buf="data/FaceTracker/Processed/CSV/au9.csv", index=False)

