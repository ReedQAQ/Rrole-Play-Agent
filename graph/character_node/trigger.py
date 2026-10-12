"""
## Acknowledgements

This project adapts parts of PersonaForge,
an ACL 2026 Findings project by Jizhou Tong and Sirui Zou.

Original repository:
https://github.com/fQwQf/PersonaForge

Licensed under the Apache License 2.0.


"""



def is_critical_interaction(self,
                                action_detail: str,
                                other_role_info: Optional[Dict[str, Any]] = None,
                                personality_profile: Optional[PersonalityProfile] = None,
                                relationship_map: Optional[Dict[str, Any]] = None) -> bool:
        """
        判断是否为关键交互（需要启用双重思维链）
        
        判断标准：
        1. 首次对话（relationship_map中无该角色记录）
        2. 涉及情感/冲突/重要决策（使用LLM分析）
        3. 话题与Agent核心兴趣高度相关
        4. 关系亲密度可能发生显著变化
        
        Args:
            action_detail: 行动详情
            other_role_info: 其他角色信息
            personality_profile: 人格画像
            relationship_map: 关系映射
        
        Returns:
            是否为关键交互
        """
        # 标准1：首次对话
        if relationship_map and other_role_info:
            other_role_code = other_role_info.get("role_code")
            if other_role_code and other_role_code not in relationship_map:
                return True
        
        # 标准2：情感强度分析（扩展版，包括压力场景关键词）
        emotional_keywords = {
            "zh": [
                # 情感类
                "喜欢", "讨厌", "生气", "开心", "难过", "愤怒", "失望", "惊喜", "爱", "恨",
                # 压力/冲突类 (用于长对话扰动场景)
                "查抄", "紧张", "咳血", "病", "婚事", "离开", "死", "走", "担心", "委屈",
                "失望", "议论", "指责", "质疑", "威胁", "危机", "紧急", "重大", "关键"
            ],
            "en": [
                "love", "hate", "angry", "happy", "sad", "disappointed", "surprised", "excited",
                "sick", "blood", "marriage", "leave", "die", "worry", "threat", "crisis", "urgent"
            ]
        }
        keywords = emotional_keywords.get(self.language, emotional_keywords["en"])
        if any(keyword in action_detail.lower() for keyword in keywords):
            return True
        
        # 标准3：话题相关性（如果action_detail包含Agent的兴趣关键词）
        if personality_profile:
            interests = personality_profile.interests
            action_lower = action_detail.lower()
            if any(interest.lower() in action_lower for interest in interests):
                return True
        
        # 标准4：使用LLM分析情感强度（可选，更准确但增加成本）
        # 这里简化处理，实际可以调用LLM判断
        
        return False


