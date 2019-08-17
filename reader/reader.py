import os
import sys
import csv
import glob
import base64
import shutil
import numpy as np
import pandas as pd
import tensorflow as tf
from six.moves import cPickle as pickle
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


#tags_to_save is tags in tags['scalars'] where the values from events.out.tfevents are stored - use yours
tags_to_save = [
"loss",
"val_loss",
"mean_absolute_error",
"val_mean_absolute_error",
"mean_squared_error",
"val_mean_squared_error"
]

#Path to directory where logs are stored
event_file = "D:\\xas"

#Iterating through all the directories where your events.out.tfevents are stored
for subdir, dirs, files in os.walk(event_file):
    for file in files:
        #Creating new path
        new_event_file = (os.path.join(subdir, file))
        event_acc = EventAccumulator(new_event_file)
        event_acc.Reload()
        ##Creating a name for csv file merged from particular scalar_events
            #It's good to name it by number of epoch, batch size, dense etc etc - 
            #In my case it inherits name from folder where event.out where stored
        csv_merged_name = subdir[7:] ##In my case [43:]
        print(new_event_file)
        tags = event_acc.Tags()
        print(tags)

        #Creating a dir for scalars_events -> .pickle and .csv files
        output_dir = os.path.join(os.path.dirname(new_event_file), "scalar_events")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for tag in tags_to_save:
            if tag not in tags['scalars']:
                continue

            events = event_acc.Scalars(tag)
            data = np.zeros([len(events), 2], dtype=np.float32)

            for step in range(len(events)):
                data[step, 0] = float(events[step].step)
                data[step, 1] = float(events[step].value)

            #Save to .pickle and .csv file
            output_file = os.path.join(output_dir, tag.replace('/', '_') + ".pickle")
            pickle.dump(data, open(output_file, 'wb'))
            x = []
            with open(output_file, 'rb') as f: 
                x = pickle.load(f)
            csv_file = os.path.join(output_dir, tag.replace('/', '_') + ".csv")
            with open(csv_file,'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f, delimiter = ',')
                writer.writerow(["id", tag])
                for line in x:
                    writer.writerow(line)

        ##Merging created .csv's into one
        os.chdir(output_dir)  
        extension = 'csv'
        all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
        combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ], sort = False)
        combined_csv_df = pd.DataFrame(combined_csv)
        grouped = combined_csv_df.groupby(['id']).sum()
        os.chdir(subdir) 
        grouped.to_csv(os.path.join(csv_merged_name + ".csv"), index=False, encoding='utf-8-sig')

        #Deleting 'scalar_events' directory - not obligatory, but elegant.
        shutil.rmtree(output_dir)

print("Done.")






