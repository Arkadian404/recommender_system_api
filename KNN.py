import pandas as pd
from surprise import Reader, Dataset, accuracy, KNNBasic
from sqlalchemy import create_engine


class KNN:
    def __init__(self, db_url):
        self.db = create_engine(db_url)
        self.df = None
        self.knn = None
        self.products = None


    def load_data_and_train(self):
        review_query = "SELECT * FROM review"
        product_query = "SELECT * FROM product"
        reviews = pd.read_sql(review_query, self.db)
        products = pd.read_sql(product_query, self.db)

        products.rename(columns={"id": "product_id", "rating": "avg_rating", "created_at": "cre_at"}, inplace=True)
        merge = pd.merge(reviews, products, on="product_id", how="left")
        df = merge.iloc[:, 0:8]
        df = df[df.parent_id.isnull()].drop(["id", "comment", "created_at", "parent_id"], axis=1)
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(df[["user_id", "product_id", "rating"]], reader=reader)
        train_df = data.build_full_trainset()
        self.knn = KNNBasic(k=5, sim_options={"name": "cosine", "user_based": True})
        self.knn.fit(train_df)
        self.products = products
        self.df = df

    def get_top_rated_products(self, n):
        top_rated_products = self.df.groupby("product_id")["rating"].mean().sort_values(ascending=False)
        top_rated_product_ids = top_rated_products.index.values[:n]
        return top_rated_product_ids

    def get_recommendations(self, user_id: int, n: int = 10):
        self.load_data_and_train()
        top_n_recommendations = []
        if user_id not in self.df["user_id"].unique():
            top_n_product_ids = self.get_top_rated_products(n)
        else:
            user_products = self.df.query("user_id == @user_id")["product_id"].unique()
            all_products = self.df["product_id"].unique()
            products_to_predict = list(set(all_products) - set(user_products))
            user_product_ratings = [(user_id, product_id, 3) for product_id in products_to_predict]
            products_predict = self.knn.test(user_product_ratings)
            top_n_recommendations = sorted(products_predict, key=lambda x: x.est, reverse=True)[:n]
            top_n_product_ids = [int(pred.iid) for pred in top_n_recommendations]

            if not top_n_product_ids:
                top_n_product_ids = self.get_top_rated_products(n)

        product_names = self.products[self.products["product_id"].isin(top_n_product_ids)]["name"].tolist()
        for p, t in zip(product_names, top_n_recommendations):
            print(u"%s voi predict la: %s" % (p.encode("utf-8", "replace"), t.est))
        return top_n_product_ids
