# faiss.py

from PIL import Image
import faiss
import matplotlib.pyplot as plt
import math
import numpy as np 
import clip
from langdetect import detect
from googletrans import Translator

class Myfaiss:
    # The __init__ method now loads both index files
    def __init__(self, visual_bin_file: str, ocr_bin_file: str, id2img_fps, device, translater, clip_backbone="ViT-B/32"):
        print("Loading VISUAL index...")
        self.visual_index = self.load_bin_file(visual_bin_file)

        print("Loading OCR TEXT index...")
        self.ocr_index = self.load_bin_file(ocr_bin_file)

        self.id2img_fps= id2img_fps
        self.device= device
        self.model, _ = clip.load(clip_backbone, device=device)
        self.translater = Translator()

    def load_bin_file(self, bin_file: str):
        print(f"Attempting to load file: {bin_file}")
        return faiss.read_index(bin_file)
    
    def show_images(self, image_paths):
        # This function remains unchanged
        fig = plt.figure(figsize=(15, 10))
        columns = int(math.sqrt(len(image_paths)))
        rows = int(np.ceil(len(image_paths)/columns))
        for i in range(1, columns*rows +1):
          img = plt.imread(image_paths[i - 1])
          ax = fig.add_subplot(rows, columns, i)
          ax.set_title('/'.join(image_paths[i - 1].split('/')[-3:]))
          plt.imshow(img)
          plt.axis("off")
        plt.show()
        
    def image_search(self, id_query, k): 
        # This function remains unchanged, as it's a pure visual search
        query_feats = self.visual_index.reconstruct(id_query).reshape(1,-1)
        scores, idx_image = self.visual_index.search(query_feats, k=k)
        idx_image = idx_image.flatten()
        infos_query = list(map(self.id2img_fps.get, list(idx_image)))
        image_paths = [info for info in infos_query]
        return scores, idx_image, infos_query, image_paths

    #################################################################
    # --- REPLACED AND UPGRADED TEXT_SEARCH METHOD ---
    #################################################################
    def text_search(self, text, k, search_type='hybrid'):
        
        # --- 1. Encode the User's Query (Same as before) ---
        if detect(text) == 'vi':
            translated = self.translater.translate(text, dest='en')
            text = translated.text
        
        text_tokens = clip.tokenize([text]).to(self.device)
        query_features = self.model.encode_text(text_tokens).cpu().detach().numpy().astype(np.float32)
        
        # CRITICAL: Normalize the query vector for accurate searching
        faiss.normalize_L2(query_features)

        # --- 2. Perform Search Based on User's Choice ---
        print(f"Performing search with type: {search_type}")
        
        if search_type == 'visual':
            # Search only the visual index
            scores, ids = self.visual_index.search(query_features, k=k)
            unique_ids = ids.flatten()
        
        elif search_type == 'ocr':
            # Search only the OCR index
            scores, ids = self.ocr_index.search(query_features, k=k)
            unique_ids = ids.flatten()

        else: # Default to 'hybrid' search
            # --- HYBRID SEARCH WITH RANK FUSION ---
            visual_scores, visual_ids = self.visual_index.search(query_features, k=k)
            ocr_scores, ocr_ids = self.ocr_index.search(query_features, k=k)

            # Reciprocal Rank Fusion (RRF) to combine results intelligently
            final_scores = {}
            # RRF constant 'k' prevents scores from being too high for top ranks
            rrf_k = 60 

            # Process and weight the VISUAL results
            for i, image_id in enumerate(visual_ids.flatten()):
                if image_id == -1: continue # Skip invalid FAISS results
                # Give a score based on rank. Lower rank (i) = higher score.
                # Visual search is less specific, so we give it a lower weight.
                score = 1.0 / (rrf_k + i)
                final_scores[image_id] = final_scores.get(image_id, 0) + score * 1.0 # Visual Weight

            # Process and weight the OCR results
            for i, image_id in enumerate(ocr_ids.flatten()):
                if image_id == -1: continue
                # OCR results are very specific, so we give them a higher weight.
                score = 1.0 / (rrf_k + i)
                final_scores[image_id] = final_scores.get(image_id, 0) + score * 1.5 # OCR Weight

            # Sort the results by their new combined score in descending order
            sorted_ids = sorted(final_scores, key=final_scores.get, reverse=True)
            unique_ids = np.array(sorted_ids)

        # --- 3. Get Image Paths for the Final Ranked Results ---
        # Make sure to handle potential None values if an ID is invalid
        infos_query = [self.id2img_fps.get(int(id_val)) for id_val in unique_ids]
        image_paths = [info for info in infos_query if info is not None]
        valid_ids = [id_val for id_val, info in zip(unique_ids, infos_query) if info is not None]

        return None, valid_ids, infos_query, image_paths