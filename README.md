RQ-Rec: Residual Quantized Hierarchical Preference Modeling for Cross-Domain Recommendation

## Getting Started <br />

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. <br />

### Installation <br />

1. **Download the datasets (Amazon Reviews 2014):** <br />
   wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Books_5.json.gz <br />
   wget https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_CDs_and_Vinyl_5.json.gz <br />
   wget https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Movies_and_TV_5.json.gz <br />
   wget https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Clothing_Shoes_and_Jewelry_5.json.gz <br />
   wget https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Sports_and_Outdoors_5.json.gz <br />
   
2. **Our LLM Generative Summaries (optional):** <br />
   https://drive.google.com/file/d/1o2VOevkE5TT6v701xNc5nRasSDMIvWle/view?usp=sharing <br />

### Runing the Project <br />

1. **put the LLM summaries under ./data:** <br />
2. **change the name of user splition json file to train_test.json:** <br />
3. **run the code. ** <br />
   python run.py <br />
