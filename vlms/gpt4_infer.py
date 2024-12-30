from openai import OpenAI
import base64
import os

import time

list_of_api_keys = [
   os.environ['OPENAI_API_KEY'],
]
api_base = os.environ['OPENAI_API_BASE']

global api_key_idx, cnt
api_key_idx = 0
cnt = 0

os.environ['OPENAI_API_KEY'] = list_of_api_keys[api_key_idx]
api_key = list_of_api_keys[api_key_idx]

# client = OpenAI()
client = OpenAI(base_url=api_base, api_key=api_key)

def gpt_infer(vlm_type, query_prompt, temperature=0):
    while True:
        try:
            response = client.chat.completions.create(
                # model="gpt-4-vision-preview",
                model=vlm_type,
                messages=[
                    # {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query_prompt}
                ],
                temperature=temperature,
                max_tokens=1000,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"API 调用失败，错误信息: {e}")
            print("正在重试...")
            time.sleep(2)  # 延迟2秒后重试


def gpt_infer_traj_gen(vlm_type, query_prompt, history_answer, trajgen_prompt, temperature=0):
    while True:
        try:
            response = client.chat.completions.create(
                # model="gpt-4-vision-preview",
                model = vlm_type,
                messages=[
                    # {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query_prompt},
                    {"role": "user", "content": "Your answer are as following:"},
                    {"role": "user", "content": history_answer},
                    {"role": "user", "content": trajgen_prompt},
                    ],
                temperature=temperature,
                max_tokens=1000,
            )
            return response.choices[0].message.content
        except Exception as e:
                print(f"API 调用失败，错误信息: {e}")
                print("正在重试...")
                time.sleep(2)  # 延迟2秒后重试

def gpt_infer_traj_gen_check(vlm_type, query_prompt, history_answer, trajgen_prompt, res_traj, traj_gen_check_prompt, temperature=0):
    while True:
        try:
            response = client.chat.completions.create(
                # model="gpt-4-vision-preview",
                model = vlm_type,
                messages=[
                    # {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query_prompt},
                    {"role": "user", "content": "Your answer are as following:"},
                    {"role": "user", "content": history_answer},
                    {"role": "user", "content": trajgen_prompt},
                    {"role": "user", "content": "Your answer are as following:"},
                    {"role": "user", "content": res_traj},
                    {"role": "user", "content": traj_gen_check_prompt},
                    ],
                temperature=temperature,
                max_tokens=1000,
            )
            return response.choices[0].message.content
        except Exception as e:
                    print(f"API 调用失败，错误信息: {e}")
                    print("正在重试...")
                    time.sleep(2)  # 延迟2秒后重试



def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')



def api_call(img_path, prompt, model="gpt-4o", temperature=0):
    base64_image = encode_image(img_path)

    response = client.chat.completions.create(
            # model="gpt-4-vision-preview",
            model="gpt-4o",
            messages=[
                {
                
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                    
                ],
                }
            ],
            temperature=temperature,
            max_tokens=1000,
        )
    
    return response

def api_call_2(img1_path, img2_path, prompt, temperature=0):
    base64_image1 = encode_image(img1_path)
    base64_image2 = encode_image(img2_path)

    response = client.chat.completions.create(
            # model="gpt-4-vision-preview",
            model="gpt-4o",
            messages=[
                {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": """
                                Consider the following two images:
                                Image 1:                        
                                """,
                    },

                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image1}",
                            "detail": "high"
                        },
                    },

                    {
                        "type": "text",
                        "text": """
                    Image 2:                        
                    """,
                    },

                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image2}",
                            "detail": "high"
                        },
                    },

                    {
                        "type": "text",
                        "text": prompt,
                    },
                    
                ],
                }
            ],
            temperature=temperature,
            max_tokens=1000,
        )
    
    return response

def extract_answer(vision_response, summary_prompt, temperature=0):
    summary_response = client.chat.completions.create(
        # model="gpt-4",
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": summary_prompt.format(vision_response)},
        ],
        temperature=temperature,
    )

    result = ''
    for choice in summary_response.choices:
        result += choice.message.content

    return result.split("/n")[0].strip().lstrip()
   
def gpt4v_infer(query_prompt, summary_prompt, img_path):
    global api_key_idx
    try:
        os.environ['OPENAI_API_KEY'] = list_of_api_keys[api_key_idx]
        response = api_call(img_path, query_prompt)
    except:
       return -1
        
    result = ''
    for choice in response.choices:
        result += choice.message.content

    print(result)

    try:
        res = extract_answer(result, summary_prompt)
        print(res)

        return res
    except:
        return -1

def gpt4v_infer_2(query_prompt, summary_prompt, img1_path, img2_path):
    global api_key_idx
    try:
        os.environ['OPENAI_API_KEY'] = list_of_api_keys[api_key_idx]
        response = api_call_2(img1_path, img2_path, query_prompt)
    except:
       return -1
        
    result = ''
    for choice in response.choices:
        result += choice.message.content

    print(result)

    try:
        res = extract_answer(result, summary_prompt)
        return res
    except:
        return -1
    

if __name__ == "__main__":

    env_name = "softgym_ClothFoldDiagonal"
    from prompt import gpt_free_query_env_prompts, gpt_summary_env_prompts
    import pickle as pkl
    from PIL import Image
    import cv2
    query_prompt = gpt_free_query_env_prompts[env_name]
    summary_prompt = gpt_summary_env_prompts[env_name]
    # # import pdb; pdb.set_trace()
    cached_path = "/home/yufei/vlm-reward-private/exp/gpt4_two_images/softgym_ClothFoldDiagonal/2024-01-18-13-03-59/vlm_1gpt4v_two_image_rewardlearn_from_preference_H1024_L2_lr0.0005/teacher_b-1_g1_m0_s0_e0/label_smooth_0.0/schedule_0/PEBBLE_init250_unsup250_inter1000_maxfeed750_seg1_acttanh_Rlr0.0003_Rbatch50_Rupdate100_en3_sample0_large_batch10_seed2/vlm_label_set/2024-01-18-13-12-42.pkl"
    with open(cached_path, 'rb') as f:
        data = pkl.load(f)
        combined_images_list, rational_labels, vlm_labels, sa_t_1, sa_t_2, r_t_1, r_t_2 = data
    
    
    for idx in range(len(combined_images_list)):
        print("=====================================")
        print("idx: ", idx)
        print("=====================================")
        
        combined_image = combined_images_list[idx]
        h, w, c = combined_image.shape
        img1 = combined_image[:, :w//2, :]
        img2 = combined_image[:, w//2:, :]
        img1 = cv2.resize(img1, (360, 360))
        img2 = cv2.resize(img2, (360, 360))
        img1_pil = Image.fromarray(img1)
        img2_pil = Image.fromarray(img2)
        img_path_1 = "data/tmp_first.jpg"
        img_path_2 = "data/tmp_second.jpg"
        img1_pil.save(img_path_1)
        img2_pil.save(img_path_2)
        res = gpt4v_infer_2(query_prompt, summary_prompt, img_path_1, img_path_2)
        print(res)