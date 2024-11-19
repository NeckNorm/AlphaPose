from data_collection_page import data_collection_page
from data_visualization_page import data_visualization_page
import justpy as jp

if __name__ == "__main__":
    jp.Route("/visualize", data_visualization_page)
    jp.Route("/collection", data_collection_page)
    jp.justpy(websockets=False) 