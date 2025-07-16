from datasets import load_dataset, Dataset, VerificationMode
import pandas
from typing import List,Dict,Tuple
import re

class JokeDataPreprocessor:
    
    def __init__(self, min_upvotes : int = 15, min_ratio: float = 2.0, min_length: int = 50, max_length : int = 2000) -> None:
        self.min_upvotes = min_upvotes
        self.min_ratio = min_ratio
        self.min_length = min_length
        self.max_length = max_length

    def load_data(self, path : str) -> pandas.DataFrame:
        try:
            data = load_dataset(path, verification_mode=VerificationMode.NO_CHECKS)
            df = data['train'].to_pandas()
            return df
        except:
           print(f"Problem with path or dataset")
           raise
    
    def calculate_metrics(self, df: pandas.DataFrame) -> pandas.DataFrame:
        df = df.copy()
        df['ratio'] = df['upvotes'] / (df['downvotes'] + 1)
        df['net_score'] = df['upvotes'] - df['downvotes']
        df['length'] = df['joke'].str.len()
        return df
    
    def clean_text(self, text: str) -> str:
        text = re.sub(r'\s+',' ',text)
        text = text.strip()
        return text
    
    def apply_quality_filters(self, df: pandas.DataFrame) -> pandas.DataFrame:
        quality_mask = (
            (df['upvotes'] >= self.min_upvotes) &
            (df['ratio'] >= self.min_ratio) &
            (df['length'] >= self.min_length) &
            (df['length'] <= self.max_length)
        )

        filtered_df = df[quality_mask].copy()
        return filtered_df
    
    def format_for_training(self, df: pandas.DataFrame) -> List[Dict]:
        formatted_data = []
        user_prompts = [
            "Opowiedz mi polski dowcip",
            "Znasz jakiś dobry polski dowcip?", 
            "Powiedz jakiś śmieszny polski dowcip",
            "Masz dla mnie jakiś polski żart?",
            "Opowiedz dowcip po polsku"
        ]

        for i, joke in enumerate(df['joke']):
            cleaned_joke = self.clean_text(joke)

            conv = {
                "messages": [
                    {
                        "role" : "system",
                        "content" : "Jesteś pomocnym asystentem, który opowiada śmieszne polskie dowcipy. Odpowiadasz tylko dowcipem, bez dodatkowych komentarzy. Nie uzywaj wylgaryzmów"
                    },
                    {
                        "role" : "user",
                        "content" : user_prompts[i % len(user_prompts)]
                    },
                    {
                        "role" : "assistant",
                        "content" : cleaned_joke
                    }

                ]
            }

            formatted_data.append(conv)

        return formatted_data
    
    def preprocess_full_pipeline(self, path : str, save_path : str = None) -> List[Dict]:
        df = self.load_data(path)
        df = self.calculate_metrics(df)

        filtered_df = self.apply_quality_filters(df)
        training_data = self.format_for_training(filtered_df)

        if save_path:
            filtered_df.to_csv(save_path, index=False)

        return training_data