import os
import pandas as pd
from datetime import datetime


class CSVLogger:

    def __init__(self, csv_file="visits.csv"):

        self.csv_file = csv_file

        self.logged_people = set()

        if not os.path.exists(csv_file):

            df = pd.DataFrame(columns=[
                "Date",
                "Time",
                "Age",
                "Gender",
                "Status"
            ])

            df.to_csv(csv_file, index=False)
    def log(self, age, gender, status):

        if status != "Senior Citizen":
            return

        now = datetime.now()

        date = now.strftime("%Y-%m-%d")

        time = now.strftime("%H:%M:%S")

        key = (
            gender,
            age,
            date,
            now.strftime("%H:%M")
        )

        if key in self.logged_people:
            return

        self.logged_people.add(key)

        row = pd.DataFrame([{

            "Date": date,

            "Time": time,

            "Age": age,

            "Gender": gender,

            "Status": status

        }])

        row.to_csv(
            self.csv_file,
            mode="a",
            header=False,
            index=False
        )  
    def get_history(self):

        return pd.read_csv(self.csv_file)
    def statistics(self):

        df = self.get_history()

        if len(df) == 0:

            return {

                "total": 0,

                "senior": 0,

                "male": 0,

                "female": 0

            }

        return {

            "total": len(df),

            "senior": len(df),

            "male": len(df[df["Gender"] == "Male"]),

            "female": len(df[df["Gender"] == "Female"])

        }
                          