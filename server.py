"""
E-Land AI Fashion Mall – HuggingFace Spaces Backend
FastAPI server with Gemini AI integration
"""

import os, json
from pathlib import Path
from typing import Dict
from fastapi import FastAPI, Body, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()
BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = (BASE_DIR / "assets").resolve()


def _asset_search_roots():
    """로그용: assets 폴더 후보."""
    roots = []
    for r in (ASSETS_DIR, Path.cwd() / "assets", BASE_DIR / "assets"):
        try:
            roots.append(r.resolve())
        except OSError:
            continue
    seen = set()
    out = []
    for r in roots:
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

MOCK_PRODUCTS = [
    {"id": "1", "brand": "Free People", "name": "Findings Slide Sandal", "price": 198000, "originalPrice": 248000, "tag": "여름 추천", "category": "신발", "keywords": "샌들 슬라이드 여름 플로럴 캐주얼"},
    {"id": "2", "brand": "MIXXO", "name": "Gemstone Bracelet", "price": 49000, "originalPrice": 59000, "tag": "강력 추천", "category": "액세서리", "keywords": "팔찌 주얼리 골드 실버 포인트"},
    {"id": "3", "brand": "Rag & Bone", "name": "Buttoned Denim Jeans - Women", "price": 647000, "originalPrice": 729000, "tag": "AI 추천 딜", "category": "바지", "keywords": "데님 진 와이드 하이웨스트 캐주얼"},
    {"id": "4", "brand": "Gap", "name": "Essential Cotton Tee", "price": 39900, "originalPrice": 49900, "tag": "트렌딩", "category": "상의", "keywords": "티셔츠 면 기본 베이직 데일리"},
    {"id": "5", "brand": "MIXXO", "name": "Silk Ruffle Blouse", "price": 159000, "originalPrice": 189000, "tag": "베스트셀러", "category": "상의", "keywords": "블라우스 실크 러플 오피스 우아한"},
    {"id": "6", "brand": "SPAO", "name": "Relaxed Fit Hoodie", "price": 45900, "originalPrice": 59900, "tag": "가성비", "keywords": "후디 맨투맨 캐주얼 편안한 오버핏"},
    {"id": "7", "brand": "New Balance", "name": "Fresh Foam Running Shoes", "price": 129000, "originalPrice": 159000, "tag": "인기", "category": "신발", "keywords": "운동화 러닝 스니커즈 스포츠 쿠셔닝"},
    {"id": "8", "brand": "WHO.A.U", "name": "Eco Cotton Cargo Pants", "price": 79900, "originalPrice": 99000, "tag": "친환경", "category": "바지", "keywords": "카고팬츠 면 와이드 캐주얼 친환경"},
]


@app.post("/api/ai-search")
async def ai_search(payload: Dict = Body(...)):
    user_input = payload.get("input", "") or payload.get("message", "")
    persona_tone = (payload.get("personaTone") or "").strip()
    if not user_input.strip():
        return {"matchedIds": [], "message": "검색어를 입력해주세요."}

    if not GEMINI_API_KEY:
        return _fallback_search(user_input)

    try:
        import google.genai as genai
        client = genai.Client(api_key=GEMINI_API_KEY)

        product_summary = json.dumps(
            [{"id": p["id"], "name": p["name"], "brand": p["brand"], "price": p["price"],
              "category": p.get("category", ""), "keywords": p.get("keywords", "")}
             for p in MOCK_PRODUCTS], ensure_ascii=False)

        tone_block = ""
        if persona_tone:
            tone_block = f"""
[페르소나 / 말투 지시 — 반드시 준수]
{persona_tone}
위 지시에 맞춰 message 필드의 한국어 말투·어조·이모지 사용을 통일하세요.
"""

        prompt = f"""당신은 E-Land Mall의 AI 패션 스타일리스트입니다. 고객이 다음과 같이 요청했습니다:

"{user_input}"
{tone_block}
아래 상품 목록에서 고객 요청에 가장 잘 맞는 상품을 1~3개 골라주세요:
{product_summary}

다음 JSON 형식으로만 응답하세요. 다른 텍스트 없이 JSON만 출력하세요:
{{
  "matchedIds": ["id1", "id2"],
  "message": "고객에게 보여줄 자연스러운 한국어 추천 메시지 (2~3문장, 왜 이 상품을 추천하는지 설명)"
}}

규칙:
- message는 선택된 페르소나의 말투 지시가 있으면 그에 맞게, 없으면 친근하고 전문적인 스타일리스트 톤으로 작성
- 추천 상품의 브랜드명과 상품명을 자연스럽게 언급
- 가격 대비 가치나 스타일링 팁을 포함
- 매칭되는 상품이 없으면 가장 인기 있는 상품을 추천하되, 솔직하게 안내"""

        response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
        text = response.text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        result = json.loads(text)
        matched_ids = result.get("matchedIds", [])
        message = result.get("message", "")
        if not isinstance(matched_ids, list):
            matched_ids = []
        if not message:
            message = f'"{user_input}"에 맞는 추천 아이템을 찾았어요!'
        return {"matchedIds": matched_ids, "message": message}

    except Exception as e:
        print(f"[AI-SEARCH] Gemini error: {e}")
        return _fallback_search(user_input)


def _fallback_search(user_input: str) -> dict:
    query = user_input.lower()
    scored = []
    for p in MOCK_PRODUCTS:
        score = 0
        searchable = f"{p['name']} {p['brand']} {p.get('keywords', '')} {p.get('category', '')}".lower()
        for word in query.split():
            if word in searchable:
                score += 2
        ko_map = {
            "바지": ["jeans", "pants", "cargo", "바지", "데님", "팬츠"],
            "상의": ["tee", "blouse", "hoodie", "상의", "셔츠", "니트"],
            "신발": ["shoes", "sandal", "running", "신발", "스니커즈", "샌들"],
            "저렴": ["가성비"], "여름": ["여름", "샌들"], "캐주얼": ["캐주얼", "데일리"],
            "오피스": ["오피스", "블라우스"], "운동": ["러닝", "스포츠", "운동화"],
        }
        for ko, matches in ko_map.items():
            if ko in query:
                for m in matches:
                    if m in searchable:
                        score += 1
        scored.append((p, score))
    scored.sort(key=lambda x: -x[1])
    top = [s for s in scored if s[1] > 0][:3]
    if top:
        names = ", ".join(f"{p['brand']} {p['name']}" for p, _ in top)
        return {
            "matchedIds": [p["id"] for p, _ in top],
            "message": f'"{user_input}"에 어울리는 아이템을 찾았어요! {names}을(를) 추천드립니다. 이미지를 클릭해서 상세 정보와 가격 비교를 확인해 보세요.'
        }
    else:
        return {
            "matchedIds": ["4", "6", "7"],
            "message": f'정확한 매치를 찾지 못했지만, 지금 가장 인기 있는 아이템들을 모아봤어요! Gap Essential Cotton Tee는 어떤 스타일에도 잘 어울리는 베이직 아이템이에요.'
        }


@app.post("/api/analyze-item")
async def analyze_item(payload: Dict = Body(...)):
    """Analyze a clothing item image with Gemini Vision for material/care info."""
    image_url = payload.get("imageUrl", "")
    item_name = payload.get("name", "")
    item_brand = payload.get("brand", "")
    query_type = payload.get("queryType", "material")  # material | care | similar

    if not GEMINI_API_KEY:
        return _fallback_analyze(item_name, item_brand, query_type)

    try:
        import google.genai as genai
        from google.genai import types
        client = genai.Client(api_key=GEMINI_API_KEY)

        prompts = {
            "material": f"""이 이미지는 {item_brand}의 "{item_name}" 상품입니다.
이미지를 분석해서 소재와 재질에 대해 알려주세요.

다음 형식으로 한국어로 답변하세요 (3~4문장):
1. 이미지에서 보이는 소재 특성 (질감, 광택, 두께 등)
2. 추정되는 소재 구성 (예: 면 100%, 폴리에스터 혼방 등)
3. 소재의 장단점
4. 계절감과 활용도

자연스러운 문장으로 작성하고, "~로 보여요", "~일 가능성이 높아요" 등 추정 표현을 사용하세요.""",

            "care": f"""이 이미지는 {item_brand}의 "{item_name}" 상품입니다.
이미지를 분석해서 세탁 및 관리 방법을 알려주세요.

다음 형식으로 한국어로 답변하세요 (3~4문장):
1. 이미지에서 추정되는 소재 기반 세탁 방법
2. 세탁 시 주의사항
3. 건조 및 보관 방법
4. 오래 입기 위한 관리 팁

자연스럽고 친절한 톤으로, 실용적인 조언을 중심으로 작성하세요.""",

            "similar": f"""이 이미지는 {item_brand}의 "{item_name}" 상품입니다.
이미지를 분석해서 비슷한 스타일의 상품을 추천해주세요.

다음 형식으로 한국어로 답변하세요 (3~4문장):
1. 이 상품의 핵심 스타일 특징
2. 비슷한 스타일을 찾을 때 주목할 키워드
3. 함께 매치하면 좋은 아이템 종류
4. 이 스타일의 트렌드 정보

자연스러운 스타일리스트 톤으로 작성하세요."""
        }

        prompt = prompts.get(query_type, prompts["material"])

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[
                types.Content(
                    parts=[
                        types.Part.from_uri(file_uri=image_url, mime_type="image/jpeg"),
                        types.Part.from_text(text=prompt),
                    ]
                )
            ],
        )

        return {"analysis": response.text.strip()}

    except Exception as e:
        print(f"[ANALYZE] Error: {e}")
        return _fallback_analyze(item_name, item_brand, query_type)


def _fallback_analyze(name: str, brand: str, query_type: str) -> dict:
    """Fallback analysis based on product name keywords."""
    nl = name.lower()
    is_denim = any(w in nl for w in ["denim", "jeans", "데님", "진"])
    is_leather = any(w in nl for w in ["leather", "레더", "가죽"])
    is_knit = any(w in nl for w in ["knit", "니트", "캐시미어", "cashmere", "스웻", "hoodie", "후디"])
    is_cotton = any(w in nl for w in ["cotton", "tee", "셔츠", "옥스포드", "shirt"])
    is_silk = any(w in nl for w in ["silk", "실크", "블라우스"])
    is_shoes = any(w in nl for w in ["shoes", "스니커즈", "sandal", "샌들", "running"])
    is_pants = any(w in nl for w in ["pants", "팬츠", "치노", "카고", "wide"])
    is_bag = any(w in nl for w in ["bag", "백", "토트"])

    if query_type == "material":
        if is_denim:
            return {"analysis": f"{brand} {name}은(는) 이미지에서 인디고 컬러의 데님 소재로 보여요. 면 98% + 스판덱스 2% 정도의 스트레치 데님일 가능성이 높습니다. 적당한 두께감이 있어 사계절 활용 가능하고, 입을수록 자연스러운 워싱이 생겨 빈티지한 매력이 더해져요."}
        elif is_leather:
            return {"analysis": f"{brand} {name}은(는) 이미지에서 매끈한 광택의 레더 소재로 보여요. PU 레더(인조 가죽) 또는 양가죽 소재로 추정됩니다. 레더 특유의 고급스러운 질감과 구조감이 있어 아우터로 활용도가 높고, 시간이 지날수록 부드러워지는 에이징 효과를 기대할 수 있어요."}
        elif is_knit:
            return {"analysis": f"{brand} {name}은(는) 이미지에서 부드러운 니트/스웻 소재로 보여요. 면 혼방 또는 폴리에스터 기모 소재로 추정되며, 보온성이 좋고 신축성이 뛰어나요. 가벼운 무게감 대비 따뜻해서 가을~겨울 데일리 아이템으로 활용도가 높습니다."}
        elif is_cotton:
            return {"analysis": f"{brand} {name}은(는) 이미지에서 깔끔한 면(코튼) 소재로 보여요. 면 100% 또는 면/폴리 혼방 옥스포드 직조일 가능성이 높습니다. 통기성이 우수하고 세탁이 용이하며, 착용할수록 부드러워지는 특성이 있어 사계절 기본 아이템으로 좋아요."}
        elif is_silk:
            return {"analysis": f"{brand} {name}은(는) 이미지에서 매끄럽고 광택감 있는 실크 혼방 소재로 보여요. 실크 또는 새틴 직조로 추정되며, 드레이프성이 뛰어나고 우아한 핏이 특징이에요. 가볍지만 고급스러운 느낌으로 오피스~세미포멀까지 활용 가능합니다."}
        elif is_shoes:
            return {"analysis": f"{brand} {name}은(는) 이미지에서 메쉬와 합성 소재 조합으로 보여요. 어퍼는 통기성 메쉬 + TPU 오버레이, 미드솔은 EVA 폼 쿠셔닝 소재로 추정됩니다. 가벼우면서도 지지력이 좋아 일상 활동과 가벼운 운동 모두에 적합해요."}
        elif is_bag:
            return {"analysis": f"{brand} {name}은(는) 이미지에서 PU 레더 또는 천연 가죽 소재로 보여요. 매끈한 표면 처리와 견고한 구조감이 특징이며, 내부는 폴리에스터 안감 처리가 되어있을 가능성이 높아요. 데일리부터 오피스까지 활용도가 높은 소재입니다."}
        elif is_pants:
            return {"analysis": f"{brand} {name}은(는) 이미지에서 면 혼방 치노/팬츠 소재로 보여요. 면 95% + 스판덱스 5% 정도의 구성으로, 적당한 신축성과 편안한 착용감이 특징이에요. 구김이 적고 관리가 쉬워 데일리 아이템으로 좋습니다."}
        else:
            return {"analysis": f"{brand} {name}은(는) 이미지에서 고품질 혼방 소재로 보여요. 면/폴리에스터 블렌드로 추정되며, 부드러운 촉감과 적당한 두께감이 특징이에요. 통기성과 내구성의 균형이 좋아 다양한 시즌에 활용할 수 있습니다."}

    elif query_type == "care":
        if is_denim:
            return {"analysis": f"{brand} {name}은(는) 데님 소재로 보여요. 뒤집어서 30도 이하 찬물에 단독 세탁을 권장해요. 특히 첫 3회 세탁은 물 빠짐이 있을 수 있으니 반드시 단독으로 세탁해 주세요. 건조기 사용은 수축 위험이 있어 자연 건조가 좋고, 직사광선을 피해 그늘에서 말려주세요."}
        elif is_leather:
            return {"analysis": f"{brand} {name}은(는) 레더 소재로 보여요. 물세탁은 절대 피하고, 전문 가죽 클리닝을 이용해 주세요. 비에 젖었을 경우 마른 수건으로 물기를 닦고 통풍이 잘 되는 곳에서 자연 건조하세요. 가죽 전용 크림으로 3~6개월마다 보습해주면 오래 예쁘게 입을 수 있어요."}
        elif is_knit:
            return {"analysis": f"{brand} {name}은(는) 니트/기모 소재로 보여요. 세탁망에 넣어 울 코스 또는 손세탁(30도 이하)을 권장해요. 비틀어 짜면 변형이 생기니 눌러서 물기를 제거하고, 평평하게 눕혀서 건조해 주세요. 옷걸이에 걸면 늘어날 수 있으니 접어서 보관하는 게 좋아요."}
        elif is_cotton:
            return {"analysis": f"{brand} {name}은(는) 면 소재로 보여요. 세탁기 일반 코스(40도 이하)로 세탁 가능하며, 흰색은 표백제 사용이 가능해요. 첫 세탁 시 약간의 수축이 있을 수 있으니 참고하세요. 다림질은 중온으로 하면 깔끔하게 관리할 수 있고, 건조기 사용도 가능하지만 자연 건조가 수명에 더 좋아요."}
        elif is_silk:
            return {"analysis": f"{brand} {name}은(는) 실크 계열 소재로 보여요. 반드시 드라이클리닝 또는 손세탁(냉수)을 해주세요. 세제는 중성 세제만 사용하고, 비틀지 말고 가볍게 눌러서 탈수해 주세요. 직사광선에 변색될 수 있으니 그늘에서 건조하고, 다림질은 저온에서 천을 대고 해주세요."}
        elif is_shoes:
            return {"analysis": f"{brand} {name}은(는) 메쉬/합성 소재 운동화로 보여요. 중성 세제를 풀은 미지근한 물에 부드러운 브러시로 살살 문질러 세척하세요. 세탁기 사용은 접착 부분 손상 위험이 있어 피하는 게 좋아요. 깔창은 따로 빼서 세탁하고, 신문지를 넣어 그늘에서 자연 건조해 주세요."}
        elif is_bag:
            return {"analysis": f"{brand} {name}은(는) 레더/PU 소재 가방으로 보여요. 부드러운 마른 천으로 자주 닦아주고, 오염 시 전용 클리너를 사용하세요. 보관 시에는 속에 종이를 넣어 형태를 유지하고, 먼지 주머니에 넣어 보관하는 게 좋아요. 습기와 직사광선을 피해주세요."}
        else:
            return {"analysis": f"{brand} {name}은(는) 혼방 소재로 보여요. 세탁망에 넣어 30도 이하 물로 세탁하는 것을 권장해요. 첫 세탁 시 단독 세탁이 안전하며, 건조기보다는 자연 건조를 추천합니다. 다림질은 중저온에서 해주시면 깔끔하게 관리할 수 있어요."}

    else:  # similar
        return {"analysis": f"{brand} {name}과(와) 비슷한 스타일을 찾으시려면, 같은 카테고리의 다른 브랜드 제품을 추천드려요. 비슷한 핏감과 디자인의 아이템들이 '사고 싶은 옷'에 있으니 확인해 보세요. 톤온톤 매치나 레이어링으로 새로운 코디를 시도해 보시는 것도 좋아요!"}


@app.get("/")
async def serve_index():
    return FileResponse(str(BASE_DIR / "index.html"))


@app.get("/assets/explore/{file_path:path}")
async def serve_assets_explore(file_path: str):
    """대화 탭 프레피 룩북 등 — explore/*.png (마운트보다 먼저 매칭)."""
    if not file_path or ".." in file_path.replace("\\", "/"):
        raise HTTPException(status_code=404, detail="Not found")
    root = (ASSETS_DIR / "explore").resolve()
    if not root.is_dir():
        raise HTTPException(status_code=404, detail="Not found")
    full = (root / file_path).resolve()
    try:
        full.relative_to(root)
    except ValueError:
        raise HTTPException(status_code=404, detail="Not found")
    if not full.is_file():
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(full)


@app.on_event("startup")
async def _log_asset_roots():
    for r in _asset_search_roots():
        if r.is_dir():
            n_pin = len(list(r.glob("pin-*.png")))
            n_sb = len(list(r.glob("same-brand-*.png")))
            n_explore = len(list(r.glob("explore/*.png")))
            if n_pin or n_sb or n_explore:
                print(f"[commerce] assets: {r} (pin: {n_pin}, same-brand: {n_sb}, explore: {n_explore})")
                return
    print(f"[commerce] WARNING: no assets in { _asset_search_roots() } — copy PNGs into assets/")


# /assets/pin-01.png, /assets/explore/preppy-lookbook.png 등 하위 경로 전부 서빙
# (기존 /assets/{filename} 단일 세그먼트만 허용 → explore/*.png 가 404였음)
app.mount("/assets", StaticFiles(directory=str(ASSETS_DIR)), name="assets")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
