import markdown

def render_markdown(text):
    if not text:
        return ""
    return markdown.markdown(text, extensions=["extra", "sane_lists", "smarty"])
