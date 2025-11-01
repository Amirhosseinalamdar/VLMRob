import os
import sys
import torch
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import io
import logging
from contextlib import asynccontextmanager
import gc
import argparse

# Add current directory to path for ViECap imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Global model variables
_model = None
_tokenizer = None
_encoder = None
_preprocess = None
_texts_embeddings = None
_entities_text = None
_device = None
_args = None

def load_viecap_model():
    """Load ViECap model and all required components following the original inference approach"""
    global _model, _tokenizer, _encoder, _preprocess, _texts_embeddings, _entities_text, _device, _args
    
    try:
        import clip
        from ClipCap import ClipCaptionModel
        from transformers import AutoTokenizer
        from utils import compose_discrete_prompts
        from load_annotations import load_entities_text
        from retrieval_categories import clip_texts_embeddings
        
        # Set default arguments matching the original inference
        _args = argparse.Namespace()
        _args.device = "cuda" if torch.cuda.is_available() else "cpu"
        _args.clip_model = 'ViT-B/32'
        _args.language_model = 'gpt2'
        _args.continuous_prompt_length = 10
        _args.clip_project_length = 10
        _args.temperature = 0.01
        _args.top_k = 3
        _args.threshold = 0.2
        _args.disable_all_entities = False
        _args.name_of_entities_text = 'vinvl_vgoi_entities'
        _args.prompt_ensemble = False
        _args.weight_path = './checkpoints/train_coco/coco_prefix-0014.pt'
        _args.using_hard_prompt = False
        _args.soft_prompt_first = False
        _args.only_hard_prompt = False
        _args.using_greedy_search = False
        _args.beam_width = 5
        _args.text_prompt = None
        
        _device = _args.device
        
        print(f"Loading ViECap model on {_device}...")
        
        # Load categories vocabulary for objects (following original code)
        if _args.name_of_entities_text == 'visual_genome_entities':
            _entities_text = load_entities_text(_args.name_of_entities_text, './annotations/vocabulary/all_objects_attributes_relationships.pickle', not _args.disable_all_entities)
            if _args.prompt_ensemble:
                _texts_embeddings = clip_texts_embeddings(_entities_text, f'./annotations/vocabulary/visual_genome_embedding_{_args.clip_model.replace("/", "")}_with_ensemble.pickle')
            else:
                _texts_embeddings = clip_texts_embeddings(_entities_text, f'./annotations/vocabulary/visual_genome_embedding_{_args.clip_model.replace("/", "")}.pickle')
        elif _args.name_of_entities_text == 'coco_entities':
            _entities_text = load_entities_text(_args.name_of_entities_text, './annotations/vocabulary/coco_categories.json', not _args.disable_all_entities)
            if _args.prompt_ensemble:
                _texts_embeddings = clip_texts_embeddings(_entities_text, f'./annotations/vocabulary/coco_embeddings_{_args.clip_model.replace("/", "")}_with_ensemble.pickle')
            else:
                _texts_embeddings = clip_texts_embeddings(_entities_text, f'./annotations/vocabulary/coco_embeddings_{_args.clip_model.replace("/", "")}.pickle')
        elif _args.name_of_entities_text == 'open_image_entities':
            _entities_text = load_entities_text(_args.name_of_entities_text, './annotations/vocabulary/oidv7-class-descriptions-boxable.csv', not _args.disable_all_entities)
            if _args.prompt_ensemble:
                _texts_embeddings = clip_texts_embeddings(_entities_text, f'./annotations/vocabulary/open_image_embeddings_{_args.clip_model.replace("/", "")}_with_ensemble.pickle')
            else:
                _texts_embeddings = clip_texts_embeddings(_entities_text, f'./annotations/vocabulary/open_image_embeddings_{_args.clip_model.replace("/", "")}.pickle')
        elif _args.name_of_entities_text == 'vinvl_vg_entities':
            _entities_text = load_entities_text(_args.name_of_entities_text, './annotations/vocabulary/VG-SGG-dicts-vgoi6-clipped.json', not _args.disable_all_entities)
            if _args.prompt_ensemble:
                _texts_embeddings = clip_texts_embeddings(_entities_text, f'./annotations/vocabulary/vg_embeddings_{_args.clip_model.replace("/", "")}_with_ensemble.pickle')
            else:
                _texts_embeddings = clip_texts_embeddings(_entities_text, f'./annotations/vocabulary/vg_embeddings_{_args.clip_model.replace("/", "")}.pickle')
        elif _args.name_of_entities_text == 'vinvl_vgoi_entities':
            _entities_text = load_entities_text(_args.name_of_entities_text, './annotations/vocabulary/vgcocooiobjects_v1_class2ind.json', not _args.disable_all_entities)
            if _args.prompt_ensemble:
                _texts_embeddings = clip_texts_embeddings(_entities_text, f'./annotations/vocabulary/vgoi_embeddings_{_args.clip_model.replace("/", "")}_with_ensemble.pickle')
            else:
                _texts_embeddings = clip_texts_embeddings(_entities_text, f'./annotations/vocabulary/vgoi_embeddings_{_args.clip_model.replace("/", "")}.pickle')
        else:
            raise RuntimeError('The entities text should be input correctly!')
        
        # Load model components
        clip_hidden_size = 640 if 'RN' in _args.clip_model else 512
        _tokenizer = AutoTokenizer.from_pretrained(_args.language_model)
        _model = ClipCaptionModel(_args.continuous_prompt_length, _args.clip_project_length, clip_hidden_size, gpt_type=_args.language_model)
        _model.load_state_dict(torch.load(_args.weight_path, map_location=_device), strict=False)
        _model.to(_device)
        _encoder, _preprocess = clip.load(_args.clip_model, device=_device)
        
        print("ViECap model and all components loaded successfully!")
        return True
        
    except Exception as e:
        print(f"Error loading ViECap model: {e}")
        import traceback
        traceback.print_exc()
        return False

def unload_viecap_model():
    """Unload model and clear GPU memory"""
    global _model, _tokenizer, _encoder, _preprocess, _texts_embeddings, _entities_text, _args
    
    if _model is not None:
        del _model
        _model = None
    if _tokenizer is not None:
        del _tokenizer
        _tokenizer = None
    if _encoder is not None:
        del _encoder
        _encoder = None
    
    _preprocess = None
    _texts_embeddings = None
    _entities_text = None
    _args = None
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    print("ViECap model unloaded and GPU memory cleared")

def generate_caption(image_pil):
    """Generate caption using the exact same approach as the original inference code"""
    global _model, _tokenizer, _encoder, _preprocess, _texts_embeddings, _entities_text, _device, _args
    
    if _model is None:
        raise RuntimeError("Model not loaded")
    
    try:
        from utils import compose_discrete_prompts
        from retrieval_categories import image_text_simiarlity, top_k_categories
        from search import greedy_search, beam_search, opt_search
        
        # Preprocess image
        image = _preprocess(image_pil).unsqueeze(dim=0).to(_device)
        
        # Encode image features
        image_features = _encoder.encode_image(image).float()
        image_features /= image_features.norm(2, dim=-1, keepdim=True)
        
        # Get continuous embeddings
        continuous_embeddings = _model.mapping_network(image_features).view(-1, _args.continuous_prompt_length, _model.gpt_hidden_size)
        
        # Handle hard prompts (following original logic)
        if _args.using_hard_prompt:
            logits = image_text_simiarlity(_texts_embeddings, temperature=_args.temperature, images_features=image_features)
            detected_objects, _ = top_k_categories(_entities_text, logits, _args.top_k, _args.threshold)
            detected_objects = detected_objects[0]  # single image inference
            
            discrete_tokens = compose_discrete_prompts(_tokenizer, detected_objects).unsqueeze(dim=0).to(_args.device)
            discrete_embeddings = _model.word_embed(discrete_tokens)
            
            if _args.only_hard_prompt:
                embeddings = discrete_embeddings
            elif _args.soft_prompt_first:
                embeddings = torch.cat((continuous_embeddings, discrete_embeddings), dim=1)
            else:
                embeddings = torch.cat((discrete_embeddings, continuous_embeddings), dim=1)
        else:
            embeddings = continuous_embeddings
        
        # Generate caption using appropriate search method
        if 'gpt' in _args.language_model:
            if not _args.using_greedy_search:
                sentences = beam_search(embeddings=embeddings, tokenizer=_tokenizer, beam_width=_args.beam_width, model=_model.gpt)
                sentence = sentences[0]  # select top 1
            else:
                sentence = greedy_search(embeddings=embeddings, tokenizer=_tokenizer, model=_model.gpt)
        else:
            sentences = opt_search(prompts=_args.text_prompt, embeddings=embeddings, tokenizer=_tokenizer, beam_width=_args.beam_width, model=_model.gpt)
            sentence = sentences[0]
        
        return sentence.strip()
    
    except Exception as e:
        raise RuntimeError(f"Caption generation failed: {str(e)}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load model
    ckpt_dir = os.getenv("VIECAP_CKPT_DIR")
    
    if not load_viecap_model():
        raise RuntimeError("Failed to load ViECap model")
    
    yield
    
    # Shutdown: unload model
    unload_viecap_model()

app = FastAPI(lifespan=lifespan, title="ViECap Server")

@app.post("/generate")
async def generate_caption_endpoint(image: UploadFile = File(...)):
    try:
        # Read and process image
        if not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        image_data = await image.read()
        image_pil = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # Generate caption using the exact same inference approach
        caption = generate_caption(image_pil)
        
        return JSONResponse(content={"caption": caption, "status": "success"})
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.get("/health")
async def health_check():
    model_status = "loaded" if _model is not None else "not loaded"
    return JSONResponse(content={
        "status": "healthy", 
        "model_loaded": _model is not None,
        "model_status": model_status,
        "device": str(_device) if _device else "unknown"
    })

@app.post("/shutdown")
async def shutdown():
    """Explicitly shutdown and unload model"""
    unload_viecap_model()
    return JSONResponse(content={"status": "shutdown", "message": "Model unloaded"})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")