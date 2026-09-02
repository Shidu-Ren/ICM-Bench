"""Structured schemas for theme, cast, timeline, and scene planning."""

from typing import List, Optional
from pydantic import BaseModel, Field


# ============================================
# 用户输入 Schema
# ============================================

class UserPrompt(BaseModel):
    """用户输入的原始 Prompt"""
    prompt: str = Field(
        description="Natural-language description of the requested video-series theme."
    )
    total_photos: int = Field(default=20, description="需要生成的照片总数")
    time_span: str = Field(default="1_year", description="相册时间跨度: 1_month, 6_months, 1_year, 3_years, 5_years")
    start_date: str = Field(default="2025-01-01", description="相册开始日期，格式: YYYY-MM-DD")


# ============================================
# 01. 主角 Schema (Protagonist)
# ============================================

class BasicInfo(BaseModel):
    """主角基本信息"""
    name_en: str = Field(description="英文名字")
    name_cn: Optional[str] = Field(default=None, description="Chinese-language name when applicable.")
    age: int = Field(description="年龄")
    gender: str = Field(description="性别: male/female")
    ethnicity: str = Field(description="种族/民族，如 East Asian, Caucasian 等")
    occupation: str = Field(description="职业详细描述")
    location: str = Field(description="居住地，如 Shanghai, China")
    marital_status: str = Field(description="婚姻状况")


class PhysicalAppearance(BaseModel):
    """外貌特征"""
    detailed_description: str = Field(
        description="详细的外貌描述（150-200词），包括体型、发型、面部特征、眼镜配饰、典型表情等。这是生成图像的关键！"
    )
    height: str = Field(description="身高，如 175cm")
    build: str = Field(description="体型，如 average, athletic, slim 等")
    hairstyle: str = Field(description="发型描述")
    has_bangs: Optional[bool] = Field(default=None, description="是否有刘海（仅女性需要填写）：true表示有刘海，false表示无刘海")
    facial_features: str = Field(description="面部特征")
    distinctive_marks: Optional[str] = Field(default=None, description="明显特征（如疤痕、纹身等）")


class ClothingStyle(BaseModel):
    """服装风格"""
    casual_daily: str = Field(description="日常休闲装的详细描述")
    work_attire: str = Field(description="工作装的详细描述")
    outdoor_activities: str = Field(description="户外活动装")
    travel_outfit: str = Field(description="旅行装")
    formal_wear: Optional[str] = Field(default=None, description="正式场合服装")


class FamilyMember(BaseModel):
    """家庭成员"""
    relation: str = Field(description="关系: spouse, son, daughter, father, mother 等")
    name_en: str = Field(description="英文名")
    age: int = Field(description="年龄")
    gender: str = Field(description="性别")
    brief_description: str = Field(description="简要描述（外貌、性格等）")


class ReferencePhoto(BaseModel):
    """参考照片描述"""
    photo_type: str = Field(description="照片类型: portrait_front_smile, casual_full_body, professional_setting 等")
    prompt: str = Field(description="Complete English image-generation prompt.")
    aspect_ratio: str = Field(default="1:1", description="宽高比: 1:1, 3:4, 16:9 等")


class Lifestyle(BaseModel):
    """生活方式描述"""
    daily_routine: str = Field(description="日常作息")
    work_life: str = Field(description="工作生活")
    weekend_activities: str = Field(description="周末活动")
    social_life: str = Field(description="社交生活")
    hobbies_details: str = Field(description="兴趣爱好详情")


class ProtagonistData(BaseModel):
    """主角完整数据"""
    basic_info: BasicInfo
    appearance: PhysicalAppearance
    clothing_styles: ClothingStyle
    family: List[FamilyMember] = Field(default_factory=list)
    personality_traits: List[str] = Field(description="性格特征列表")
    hobbies: List[str] = Field(description="爱好列表")
    lifestyle: Lifestyle = Field(description="生活方式描述")
    reference_photos: List[ReferencePhoto] = Field(description="参考照片描述；视频流程通常只使用1张正面参考图")


# ============================================
# 02. 分布计划 Schema (Distribution)
# ============================================

class SceneTypeDistribution(BaseModel):
    """按场景类型分布"""
    home: int = Field(description="家庭场景照片数")
    work: int = Field(description="工作场景照片数")
    social: int = Field(description="社交场景照片数")
    travel: int = Field(description="旅行场景照片数")
    outdoor: int = Field(default=0, description="户外场景照片数")
    other: int = Field(default=0, description="其他场景照片数")


class PhotoTypeDistribution(BaseModel):
    """按拍摄类型分布"""
    selfie: int = Field(description="自拍照片数")
    group_photo: int = Field(description="合照数")
    candid: int = Field(description="抓拍照片数")
    portrait: int = Field(description="肖像照片数")
    event: int = Field(default=0, description="活动照片数")


class TimePeriodDistribution(BaseModel):
    """按时间段分布"""
    morning: int = Field(description="早晨照片数")
    afternoon: int = Field(description="下午照片数")
    evening: int = Field(description="傍晚照片数")
    night: int = Field(description="夜晚照片数")


class SeasonDistribution(BaseModel):
    """按季节分布"""
    spring: int = Field(description="春季照片数")
    summer: int = Field(description="夏季照片数")
    autumn: int = Field(description="秋季照片数")
    winter: int = Field(description="冬季照片数")


class PhotoDistribution(BaseModel):
    """照片分布统计"""
    by_scene_type: SceneTypeDistribution = Field(description="按场景类型分布")
    by_photo_type: PhotoTypeDistribution = Field(description="按拍摄类型分布")
    by_time_period: TimePeriodDistribution = Field(description="按时间段分布")
    by_season: SeasonDistribution = Field(description="按季节分布")


class DistributionSummary(BaseModel):
    """分布总体统计"""
    total_photos: int = Field(description="总照片数")
    time_span_days: int = Field(description="时间跨度（天）")
    video_count: int = Field(default=0, description="视频数量")


class Milestone(BaseModel):
    """重要时间节点"""
    date: str = Field(description="日期 YYYY-MM-DD")
    event: str = Field(description="事件描述")
    photos: int = Field(description="照片数量")


class DistributionPlan(BaseModel):
    """分布计划"""
    summary: DistributionSummary = Field(description="总体统计")
    distribution: PhotoDistribution = Field(description="照片分布详情")
    milestones: List[Milestone] = Field(description="重要时间节点")


# ============================================
# 03. 人物组 Schema (Character Groups)
# ============================================

class CharacterReference(BaseModel):
    """人物参考信息"""
    id: str = Field(description="人物唯一ID，如 char_001")
    name_en: str = Field(description="英文名")
    name_cn: Optional[str] = Field(default=None, description="中文名（如果适用）")
    relation_to_protagonist: str = Field(description="与主角的关系")
    age: int
    gender: str = Field(description="性别：male 或 female")
    has_bangs: Optional[bool] = Field(default=None, description="是否有刘海（仅女性需要填写）：true表示有刘海，false表示无刘海，男性填null")
    appearance_description: str = Field(description="外貌详细描述（用于生成人物参考图），必须包含强区分特征避免复制人现象")
    typical_clothing: str = Field(description="典型服装描述")
    wardrobe_options: List[str] = Field(
        default_factory=list,
        description="2-3 base reusable outfit descriptions for this adult character; the video workflow may expand them to at most five global outfit slots after batch outlining"
    )
    personality_brief: str = Field(description="性格简述")
    reference_photos: List[ReferencePhoto] = Field(description="1个正面参考照片描述")


class CharacterGroupsSummary(BaseModel):
    """人物组统计信息"""
    total_characters: int = Field(description="总人物数")
    core_family_count: int = Field(default=0, description="核心家庭成员数")
    close_friends_count: int = Field(default=0, description="亲密朋友数")
    colleagues_count: int = Field(default=0, description="同事数")
    other_count: int = Field(default=0, description="其他人物数")


class CharacterGroupsData(BaseModel):
    """按组分类的人物"""
    core_family: List[CharacterReference] = Field(default_factory=list, description="核心家庭成员")
    close_friends: List[CharacterReference] = Field(default_factory=list, description="亲密朋友")
    colleagues: List[CharacterReference] = Field(default_factory=list, description="同事")
    acquaintances: List[CharacterReference] = Field(default_factory=list, description="熟人")
    other: List[CharacterReference] = Field(default_factory=list, description="其他人物")


class CharacterGroups(BaseModel):
    """人物组"""
    character_groups: CharacterGroupsData = Field(description="按组分类的人物")
    summary: CharacterGroupsSummary = Field(description="统计信息")


# ============================================
# 04. 场景组 Schema (Scene Groups)
# ============================================

class SceneReference(BaseModel):
    """场景参考信息"""
    id: str = Field(description="场景唯一ID，如 scene_001")
    name_en: str = Field(description="场景英文名，如 Living Room, Office Desk")
    name_cn: Optional[str] = Field(default=None, description="Chinese-language scene name when applicable.")
    category: str = Field(description="场景类别: home, work, social, travel, outdoor 等")
    frequency: str = Field(description="出现频率: high, medium, low")
    description: str = Field(description="场景详细描述（用于背景生成）")
    lighting: str = Field(description="光照条件: natural daylight, warm indoor, night lighting 等")
    mood: str = Field(description="氛围: cozy, professional, lively, serene 等")
    background_prompt: str = Field(description="完整的背景图生成 prompt（英文）")


class SceneGroupsData(BaseModel):
    """按频率分类的场景"""
    high_frequency: List[SceneReference] = Field(default_factory=list, description="高频场景")
    medium_frequency: List[SceneReference] = Field(default_factory=list, description="中频场景")
    low_frequency: List[SceneReference] = Field(default_factory=list, description="低频场景")


class SceneGroupsSummary(BaseModel):
    """场景组统计信息"""
    total_scenes: int = Field(description="总场景数")
    high_frequency_count: int = Field(description="高频场景数")
    medium_frequency_count: int = Field(description="中频场景数")
    low_frequency_count: int = Field(description="低频场景数")


class SceneGroups(BaseModel):
    """场景组"""
    scene_groups: SceneGroupsData = Field(description="按频率分类的场景")
    summary: SceneGroupsSummary = Field(description="统计信息")
