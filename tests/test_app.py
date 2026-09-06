from streamlit.testing.v1 import AppTest

def test_demo_submit_and_approve():
    app=AppTest.from_file("app.py",default_timeout=20).run()
    assert not app.exception
    app.button[0].click().run()
    assert not app.exception
    assert app.session_state.graph.get_state(app.session_state.config).values["status"] == "pending_review"
    app.text_input[0].set_value("REVISOR-TESTE")
    app.checkbox[0].check()
    app.button[1].click().run()
    assert not app.exception
    assert app.session_state.graph.get_state(app.session_state.config).values["status"] == "approved"
