import os
import json
import base64
import time
import re
import math
from typing import Tuple, Optional, Dict, Any, List
from openai import OpenAI
from tqdm import tqdm
from datetime import datetime

DATASET_ROOT = r""
RAW_DATA_BASE_PATH = r""
TARGET_FILE = "Image_Only.json"
OUTPUT_FILE = f"eval_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")
CLIENT_TIMEOUT = 60
MAX_RETRY = 3

client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL, timeout=CLIENT_TIMEOUT)

def load_metadata(root_dir: str) -> str:
    path = os.path.join(root_dir, "dataset_metadata.json")
    if not os.path.exists(path):
        return ""
    
    with open(path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    
    text = "\n[Domain Knowledge / Metadata]\n"
    
    sat_data = meta.get("Satellite_Metadata", {})
    text += "1. Satellite Sensor Specifications:\n"
    for sat, channels in sat_data.items():
        text += f"   - {sat}:\n"
        sorted_chs = sorted(channels.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 999)
        for ch_id, specs in sorted_chs:
            spec_str = ", ".join([f"{k}={v}" for k, v in specs.items()])
            text += f"     * Band {ch_id}: {spec_str}\n"
            
    st_data = meta.get("Station_Metadata", {})
    text += "\n2. Station Data Definitions:\n"
    for k, v in st_data.get("Descriptions", {}).items():
        text += f"   - {k}: {v}\n"
    
    text += "\n3. Field Abbreviations:\n"
    for k, v in st_data.get("Field_Definitions", {}).items():
        text += f"   - {k} represents: {', '.join(v)}\n"
        
    return text

def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def prepare_image_messages(image_rel_paths: str, base_path: str) -> List[Dict[str, Any]]:
    if not image_rel_paths:
        return []
    
    paths = [p.strip() for p in image_rel_paths.split(',') if p.strip()]
    messages = []
    
    for p in paths:
        full_path = os.path.join(base_path, p)
        if os.path.exists(full_path):
            try:
                b64 = encode_image(full_path)
                messages.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "low"}
                })
            except Exception:
                pass
        else:
            pass
            
    return messages

def format_station_data(stations: Optional[Dict[str, Any]]) -> str:
    if not stations:
        return "No station data available."
    
    try:
        return json.dumps(stations, ensure_ascii=False, indent=None)
    except:
        return "Error parsing station data."

def normalize_text(s: str) -> str:
    if s is None: return ""
    s = s.strip().lower()
    s = re.sub(r'\s+', ' ', s)
    return s

def try_parse_numeric(s: str) -> Optional[float]:
    m = re.search(r'([-+]?\d+(?:\.\d+)?)', normalize_text(s))
    if m:
        return float(m.group(1))
    return None

def extract_final_answer(model_response: str) -> Dict[str, Any]:
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', model_response, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except:
            pass
            
    try:
        start = model_response.find('{')
        end = model_response.rfind('}')
        if start != -1 and end != -1:
            return json.loads(model_response[start:end+1])
    except:
        pass
        
    return {"final_answer": model_response.strip(), "answer_type": "text"}

def qa_instance_score(ground_truth: str, extracted: Dict[str, Any]) -> Dict[str, Any]:
    gt = str(ground_truth).strip()
    pred = str(extracted.get("final_answer", "")).strip()
    
    is_categorical = re.match(r'^[A-Za-z\s]+$', gt) and len(gt) < 30
    
    if not pred:
        return {"score": 0.0, "reason": "Empty prediction"}

    YES_SET = {"yes", "true", "是", "1"}
    NO_SET = {"no", "false", "否", "0"}
    if gt.lower() in YES_SET or gt.lower() in NO_SET:
        p_norm = pred.lower()
        p_bool = "yes" if any(x in p_norm for x in YES_SET) else ("no" if any(x in p_norm for x in NO_SET) else "unknown")
        g_bool = "yes" if gt.lower() in YES_SET else "no"
        return {"score": 1.0 if p_bool == g_bool else 0.0, "type": "boolean"}

    gt_num = try_parse_numeric(gt)
    if gt_num is not None:
        pred_num = try_parse_numeric(pred)
        if pred_num is not None:
            error = abs(pred_num - gt_num)
            tolerance = 0.5 if gt_num < 10 else (gt_num * 0.1)
            score = 1.0 if error <= tolerance else max(0.0, 1.0 - (error / (gt_num + 1e-6)))
            return {"score": score, "type": "numeric", "gt": gt_num, "pred": pred_num}
    
    if is_categorical:
        if gt.lower() == pred.lower():
            return {"score": 1.0, "type": "classification_exact"}
        if gt.lower() in pred.lower() or pred.lower() in gt.lower():
            return {"score": 0.8, "type": "classification_partial"}
            
    common = set(pred.lower().split()) & set(gt.lower().split())
    score = len(common) / max(len(gt.split()), 1)
    return {"score": min(1.0, score), "type": "text_overlap"}

def evaluate_dataset():
    dataset_path = os.path.join(DATASET_ROOT, TARGET_FILE)
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        return

    metadata_context = load_metadata(DATASET_ROOT)
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    print(f"Loaded {len(dataset)} items from {TARGET_FILE}")
    
    results = []
    
    for i, item in enumerate(tqdm(dataset)):
        question = item.get("Text", "")
        images_str = item.get("Image", "")
        stations_data = item.get("Stations", None)
        gt = str(item.get("Ground truth", ""))
        
        system_prompt = "You are an expert AI for disaster assessment from satellite imagery and station data.\n"
        if metadata_context:
            system_prompt += metadata_context
        
        user_content = []
        user_content.append({"type": "text", "text": f"Task Instruction: {question}"})
        
        if stations_data:
            st_text = format_station_data(stations_data)
            user_content.append({"type": "text", "text": f"Station Data Context:\n{st_text}"})
            
        user_content.append({"type": "text", "text": 
            "\nOutput Requirement:\n"
            "Please analyze the inputs and provide the answer.\n"
            "You MUST output your final answer in a strict JSON format:\n"
            "```json\n"
            "{\"final_answer\": \"YOUR_ANSWER_HERE\", \"answer_type\": \"boolean/numeric/class/text\"}\n"
            "```\n"
            "For numeric answers, output only the number.\n"
            "For classification, output the class name."
        })
        
        img_msgs = prepare_image_messages(images_str, RAW_DATA_BASE_PATH)
        user_content.extend(img_msgs)
        
        raw_resp = ""
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.1,
                max_tokens=300
            )
            raw_resp = resp.choices[0].message.content
            extracted = extract_final_answer(raw_resp)
        except Exception as e:
            print(f"API Error: {e}")
            extracted = {"final_answer": "API Error"}

        score_info = qa_instance_score(gt, extracted)
        
        result_item = {
            "id": item.get("Question_id"),
            "task": item.get("Task"),
            "subtask": item.get("Subtask"),
            "ground_truth": gt,
            "prediction": extracted.get("final_answer"),
            "score": score_info["score"],
            "score_type": score_info.get("type"),
            "raw_response": raw_resp
        }
        results.append(result_item)
        
        if i < 3: 
            print(f"\n[Sample {i}] GT: {gt} | Pred: {extracted.get('final_answer')} | Score: {score_info['score']}")

    total_score = sum(r["score"] for r in results)
    avg_score = total_score / len(results) if results else 0
    
    task_stats = {}
    for r in results:
        t = r.get("task", "Unknown")
        if t not in task_stats: task_stats[t] = []
        task_stats[t].append(r["score"])
        
    print("\n" + "="*40)
    print(f"Evaluation Complete. Overall Score: {avg_score:.4f}")
    print("Scores by Task:")
    for t, scores in task_stats.items():
        print(f"  - {t}: {sum(scores)/len(scores):.4f} (n={len(scores)})")
    print("="*40)
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump({"summary": {"overall": avg_score, "by_task": {k: sum(v)/len(v) for k,v in task_stats.items()}}, "details": results}, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    evaluate_dataset()