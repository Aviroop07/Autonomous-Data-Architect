import pytest
from unittest.mock import patch
from src.pipeline.stage2.agents.conceptual_extractor.agent import extract_conceptual_model
from src.pipeline.stage2.agents.conceptual_verifier.agent import verify_conceptual_model
from src.pipeline.stage2.mapper.conceptual_model import ConceptualModel, Entity, CMAttribute
from src.pipeline.stage2.models.conceptual_critique import ConceptualCritiqueReport
from src.pipeline.stage2.models.data_types import DataType

@pytest.mark.anyio
async def test_extract_conceptual_model():
    mock_model = ConceptualModel(
        entities=[Entity(name="Mock", attributes=[CMAttribute(name="mock_attr", type=DataType.VARCHAR)])]
    )
    
    with patch("src.pipeline.stage2.agents.conceptual_extractor.agent.get_response") as mock_get:
        mock_get.return_value = (mock_model, 100)
        
        parsed, tokens = await extract_conceptual_model("facts", "query")
        
        assert parsed == mock_model
        assert tokens == 100
        mock_get.assert_called_once()

@pytest.mark.anyio
async def test_verify_conceptual_model():
    mock_critique = ConceptualCritiqueReport(is_valid=True)
    mock_model = ConceptualModel(entities=[])
    
    with patch("src.pipeline.stage2.agents.conceptual_verifier.agent.get_response") as mock_get:
        mock_get.return_value = (mock_critique, 50)
        
        parsed, tokens = await verify_conceptual_model("facts", mock_model)
        
        assert parsed == mock_critique
        assert tokens == 50
        mock_get.assert_called_once()
