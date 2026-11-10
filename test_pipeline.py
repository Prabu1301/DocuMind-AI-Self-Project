from rag_pipeline import run_rag_simulation

def test_ai_pipeline():
    """
    Automated test to validate that the local RAG simulation 
    executes successfully without throwing exceptions.
    """
    try:
        run_rag_simulation()
        assert True
    except Exception as e:
        assert False, f"Pipeline crashed with error: {e}"