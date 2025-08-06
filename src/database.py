"""
Lobbying Effects Analysis Model Classes

This module contains two main classes:
1. DataBase: Responsible for loading and treating panel data
2. LobbyingEffectsModel: Responsible for running econometric models with different topics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# Set display options
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 100)
plt.style.use("seaborn-v0_8")


class DataBase:
    """
    Class responsible for loading and treating panel data for lobbying effects analysis.
    """

    def __init__(self, data_path="./data/gold/"):
        """
        Initialize DataBase with path to data files.

        Args:
            data_path (str): Path to the data directory
        """
        self.data_path = data_path
        self.df: pd.DataFrame = pd.DataFrame()
        self.df_filtered: pd.DataFrame = pd.DataFrame()
        self.column_sets: dict[str, list[str]] = {}

    def load_data(self, time_frequency="monthly"):
        """
        Load panel data from CSV files.

        Args:
            time_frequency (str): Either "monthly" or "weekly"
        """
        print("=== Loading Data ===")

        if time_frequency == "monthly":
            suffix = "Ym"
        elif time_frequency == "weekly":
            suffix = "Yw"
        else:
            raise ValueError("time_frequency must be 'monthly' or 'weekly'")

        # Load data
        df_meetings = pd.read_csv(
            f"{self.data_path}/panel_data_meetings_{suffix}_v020701.csv"
        )
        df_meps = pd.read_csv(f"{self.data_path}/panel_data_meps_{suffix}_v020702.csv")
        df_questions = pd.read_csv(
            f"{self.data_path}/panel_data_questions_{suffix}_v020703.csv"
        )
        df_graph = pd.read_csv(
            f"{self.data_path}/panel_data_graph_{suffix}_v020704.csv"
        )

        # Clean data
        del df_meetings["Y-w"]
        del df_meps["Y-w"]
        del df_questions["Y-w"]

        # Convert time column to datetime
        time_col = "Y-m" if time_frequency == "monthly" else "Y-w"
        df_meetings[time_col] = pd.to_datetime(df_meetings[time_col], format="%Y-%m")
        df_meps[time_col] = pd.to_datetime(df_meps[time_col], format="%Y-%m")
        df_questions[time_col] = pd.to_datetime(df_questions[time_col], format="%Y-%m")
        df_graph[time_col] = pd.to_datetime(df_graph[time_col], format="%Y-%m")

        # Set indices
        df_meetings.set_index(["member_id", time_col], inplace=True)
        df_meps.set_index(["member_id", time_col], inplace=True)
        df_questions.rename(columns={"creator": "member_id"}, inplace=True)
        df_questions.set_index(["member_id", time_col], inplace=True)
        df_graph.set_index(["member_id", time_col], inplace=True)

        # Add prefixes
        df_questions_prefixed = df_questions.add_prefix("questions_")
        df_meps_prefixed = df_meps.add_prefix("meps_")
        df_meetings_prefixed = df_meetings.add_prefix("meetings_")
        df_graph_prefixed = df_graph.add_prefix("graph_")

        # Join all dataframes
        self.df = (
            df_meps_prefixed.join(
                df_questions_prefixed, on=["member_id", time_col], how="left"
            )
            .join(df_meetings_prefixed, on=["member_id", time_col], how="left")
            .join(df_graph_prefixed, on=["member_id", time_col], how="left")
        )

        # Fill missing values
        self.df.fillna(0, inplace=True)

        print(f"Data loaded successfully. Shape: {self.df.shape}")

    def filter_time_period(self, start_date="2019-07", end_date="2024-11"):
        """
        Filter data to specific time period.

        Args:
            start_date (str): Start date in format "YYYY-MM"
            end_date (str): End date in format "YYYY-MM"
        """
        print("=== Filtering Time Period ===")

        self.df_filtered = self.df.loc[
            (self.df.index.get_level_values(1) > pd.to_datetime(start_date))
            & (self.df.index.get_level_values(1) < pd.to_datetime(end_date))
        ]

        print(f"Filtered data shape: {self.df_filtered.shape}")
        print(
            f"Time period: {self.df_filtered.index.get_level_values(1).min()} to {self.df_filtered.index.get_level_values(1).max()}"
        )
        print(f"Number of MEPs: {self.df_filtered.index.get_level_values(0).nunique()}")

    def create_log_transformations(self):
        """
        Create log transformations for all relevant variables.
        """
        print("=== Creating Log Transformations ===")
        # Create log transformations for specific sets

        sets_to_iterate = [
            "QUESTIONS_TOPICS_COLUMNS",
            "MEETINGS_TOPICS_COLUMNS",
            "MEETINGS_MEMBER_CAPACITY_COLUMNS",
            "MEETINGS_CATEGORY_COLUMNS",
        ]

        for set_name in sets_to_iterate:
            to_iterate = self.column_sets[set_name].copy()
            for c in to_iterate:
                new_col = f"log_{c}"
                self.df_filtered[new_col] = np.log(self.df_filtered[c] + 1)
                self.column_sets[set_name].remove(c)
                self.column_sets[set_name].append(new_col)

        print("Log transformations created successfully.")

    def _define_column_sets(self):
        """
        Define comprehensive column sets for different variable categories.
        """
        column_sets = {}

        # Questions topics
        column_sets["QUESTIONS_TOPICS_COLUMNS"] = [
            c
            for c in self.df.columns
            if "questions_infered_topic" in c and "log_" not in c
        ]

        # Meetings topics
        column_sets["MEETINGS_TOPICS_COLUMNS"] = [
            "meetings_l_agriculture",
            "meetings_l_economics_and_trade",
            "meetings_l_education",
            "meetings_l_environment_and_climate",
            "meetings_l_foreign_and_security_affairs",
            "meetings_l_health",
            "meetings_l_human_rights",
            "meetings_l_infrastructure_and_industry",
            "meetings_l_technology",
        ]

        # Meetings member capacity
        column_sets["MEETINGS_MEMBER_CAPACITY_COLUMNS"] = [
            "meetings_member_capacity_Committee chair",
            "meetings_member_capacity_Delegation chair",
            "meetings_member_capacity_Member",
            "meetings_member_capacity_Rapporteur",
            "meetings_member_capacity_Rapporteur for opinion",
            "meetings_member_capacity_Shadow rapporteur",
            "meetings_member_capacity_Shadow rapporteur for opinion",
        ]

        # Meetings category
        column_sets["MEETINGS_CATEGORY_COLUMNS"] = [
            "meetings_l_category_Business",
            "meetings_l_category_NGOs",
            "meetings_l_category_Other",
            "meetings_l_budget_cat_lower",
            "meetings_l_budget_cat_middle",
            "meetings_l_budget_cat_upper",
            "meetings_l_days_since_registration_lower",
            "meetings_l_days_since_registration_middle",
            "meetings_l_days_since_registration_upper",
        ]

        # Meetings head office
        # column_sets["MEETINGS_HEAD_OFFICE_COLUMNS"] = [
        #     c for c in self.df_filtered.columns if "meetings_l_head_office_country" in c
        # ]

        # MEPs political groups
        column_sets["MEPS_POLITICAL_GROUP_COLUMNS"] = [
            "meps_POLITICAL_GROUP_5148.0",
            "meps_POLITICAL_GROUP_5151.0",
            "meps_POLITICAL_GROUP_5152.0",
            "meps_POLITICAL_GROUP_5153.0",
            "meps_POLITICAL_GROUP_5154.0",
            "meps_POLITICAL_GROUP_5155.0",
            "meps_POLITICAL_GROUP_5588.0",
            "meps_POLITICAL_GROUP_5704.0",
            "meps_POLITICAL_GROUP_6259.0",
            "meps_POLITICAL_GROUP_6561.0",
            "meps_POLITICAL_GROUP_7018.0",
            "meps_POLITICAL_GROUP_7028.0",
            "meps_POLITICAL_GROUP_7035.0",
            "meps_POLITICAL_GROUP_7036.0",
            "meps_POLITICAL_GROUP_7037.0",
            "meps_POLITICAL_GROUP_7038.0",
            "meps_POLITICAL_GROUP_7150.0",
            "meps_POLITICAL_GROUP_7151.0",
        ]

        # MEPs countries
        column_sets["MEPS_COUNTRY_COLUMNS"] = [
            "meps_COUNTRY_AUT",
            "meps_COUNTRY_BEL",
            "meps_COUNTRY_BGR",
            "meps_COUNTRY_CYP",
            "meps_COUNTRY_CZE",
            "meps_COUNTRY_DEU",
            "meps_COUNTRY_DNK",
            "meps_COUNTRY_ESP",
            "meps_COUNTRY_EST",
            "meps_COUNTRY_FIN",
            "meps_COUNTRY_FRA",
            "meps_COUNTRY_GBR",
            "meps_COUNTRY_GRC",
            "meps_COUNTRY_HRV",
            "meps_COUNTRY_HUN",
            "meps_COUNTRY_IRL",
            "meps_COUNTRY_ITA",
            "meps_COUNTRY_LTU",
            "meps_COUNTRY_LUX",
            "meps_COUNTRY_LVA",
            "meps_COUNTRY_MLT",
            "meps_COUNTRY_NLD",
            "meps_COUNTRY_POL",
            "meps_COUNTRY_PRT",
            "meps_COUNTRY_ROU",
            "meps_COUNTRY_SVK",
            "meps_COUNTRY_SVN",
            "meps_COUNTRY_SWE",
        ]

        # MEPs positions
        column_sets["MEPS_POSITIONS_COLUMNS"] = [
            "meps_COMMITTEE_PARLIAMENTARY_SPECIAL - CHAIR",
            # "meps_COMMITTEE_PARLIAMENTARY_SPECIAL - CHAIR_VICE",
            "meps_COMMITTEE_PARLIAMENTARY_SPECIAL - MEMBER",
            # "meps_COMMITTEE_PARLIAMENTARY_SPECIAL - MEMBER_SUBSTITUTE",
            "meps_COMMITTEE_PARLIAMENTARY_STANDING - CHAIR",
            # "meps_COMMITTEE_PARLIAMENTARY_STANDING - CHAIR_VICE",
            "meps_COMMITTEE_PARLIAMENTARY_STANDING - MEMBER",
            # "meps_COMMITTEE_PARLIAMENTARY_STANDING - MEMBER_SUBSTITUTE",
            "meps_COMMITTEE_PARLIAMENTARY_SUB - CHAIR",
            # "meps_COMMITTEE_PARLIAMENTARY_SUB - CHAIR_VICE",
            "meps_COMMITTEE_PARLIAMENTARY_SUB - MEMBER",
            # "meps_COMMITTEE_PARLIAMENTARY_SUB - MEMBER_SUBSTITUTE",
            # "meps_COMMITTEE_PARLIAMENTARY_TEMPORARY - CHAIR",
            # "meps_COMMITTEE_PARLIAMENTARY_TEMPORARY - CHAIR_VICE",
            # "meps_COMMITTEE_PARLIAMENTARY_TEMPORARY - MEMBER",
            # "meps_COMMITTEE_PARLIAMENTARY_TEMPORARY - MEMBER_SUBSTITUTE",
            # "meps_DELEGATION_JOINT_COMMITTEE - CHAIR",
            # "meps_DELEGATION_JOINT_COMMITTEE - CHAIR_VICE",
            # "meps_DELEGATION_JOINT_COMMITTEE - MEMBER",
            # "meps_DELEGATION_JOINT_COMMITTEE - MEMBER_SUBSTITUTE",
            "meps_DELEGATION_PARLIAMENTARY - CHAIR",
            # "meps_DELEGATION_PARLIAMENTARY - CHAIR_VICE",
            "meps_DELEGATION_PARLIAMENTARY - MEMBER",
            # "meps_DELEGATION_PARLIAMENTARY - MEMBER_SUBSTITUTE",
            # "meps_DELEGATION_PARLIAMENTARY_ASSEMBLY - CHAIR",
            # "meps_DELEGATION_PARLIAMENTARY_ASSEMBLY - CHAIR_VICE",
            # "meps_DELEGATION_PARLIAMENTARY_ASSEMBLY - MEMBER",
            # "meps_DELEGATION_PARLIAMENTARY_ASSEMBLY - MEMBER_SUBSTITUTE",
            "meps_EU_INSTITUTION - PRESIDENT",
            # "meps_EU_INSTITUTION - PRESIDENT_VICE",
            "meps_EU_INSTITUTION - QUAESTOR",
            "meps_EU_POLITICAL_GROUP - CHAIR",
            # "meps_EU_POLITICAL_GROUP - CHAIR_CO",
            # "meps_EU_POLITICAL_GROUP - CHAIR_VICE",
            # "meps_EU_POLITICAL_GROUP - CHAIR_VICE_FIRST",
            "meps_EU_POLITICAL_GROUP - MEMBER_BUREAU",
            # "meps_EU_POLITICAL_GROUP - PRESIDENT_CO",
            "meps_EU_POLITICAL_GROUP - TREASURER",
            "meps_EU_POLITICAL_GROUP - TREASURER_CO",
            "meps_NATIONAL_CHAMBER - PRESIDENT_VICE",
            "meps_WORKING_GROUP - CHAIR",
            # "meps_WORKING_GROUP - CHAIR_CO",
            # "meps_WORKING_GROUP - CHAIR_VICE",
            "meps_WORKING_GROUP - MEMBER",
            "meps_WORKING_GROUP - MEMBER_BUREAU",
            # "meps_WORKING_GROUP - PRESIDENT_PARLIAMENT_STOA",
        ]

        # Graph
        column_sets["GRAPH_AUTHORITY_COLUMNS"] = [
            "graph_authority",
            "graph_l_agriculture_authority_percentage",
            "graph_l_economics_and_trade_authority_percentage",
            "graph_l_education_authority_percentage",
            "graph_l_environment_and_climate_authority_percentage",
            "graph_l_foreign_and_security_affairs_authority_percentage",
            "graph_l_health_authority_percentage",
            "graph_l_human_rights_authority_percentage",
            "graph_l_infrastructure_and_industry_authority_percentage",
            "graph_l_technology_authority_percentage",
        ]

        column_sets["GRAPH_PERCENTAGE_COLUMNS"] = [
            "graph_l_agriculture_percentage",
            "graph_l_economics_and_trade_percentage",
            "graph_l_education_percentage",
            "graph_l_environment_and_climate_percentage",
            "graph_l_foreign_and_security_affairs_percentage",
            "graph_l_health_percentage",
            "graph_l_human_rights_percentage",
            "graph_l_infrastructure_and_industry_percentage",
            "graph_l_technology_percentage",
        ]

        return column_sets

    def get_data(self):
        """
        Get the processed data.

        Returns:
            tuple: (df_filtered, column_sets)
        """
        return self.df_filtered

    def filter_columns(self):
        """
        Filter columns to only include those in the column sets.
        """
        # Define column sets
        self.column_sets = self._define_column_sets()

        self.df_filtered = self.df_filtered[
            [c for v in self.column_sets.values() for c in v]
        ]

    def rename_columns(self):
        """
        Rename columns to remove white spaces and add underscores.
        """
        for set_name in self.column_sets:
            for i, c in enumerate(self.column_sets[set_name]):
                new_col = c.replace(" ", "_")
                self.df_filtered.rename(columns={c: new_col}, inplace=True)
                self.column_sets[set_name][i] = new_col

    def prepare_data(
        self, time_frequency="monthly", start_date="2019-07", end_date="2024-11"
    ):
        """
        Complete data preparation pipeline.

        Args:
            time_frequency (str): Either "monthly" or "weekly"
            start_date (str): Start date in format "YYYY-MM"
            end_date (str): End date in format "YYYY-MM"
        """
        self.load_data(time_frequency)
        self.filter_time_period(start_date, end_date)
        self.filter_columns()
        self.create_log_transformations()
        self.rename_columns()
        return self.get_data(), self.column_sets
