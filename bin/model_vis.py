from gradio_client import Client


def request_vis(alias):
    client = Client("https://a4e56818ec76fcee08.gradio.live")
    result = client.predict(
            alias=alias,
            box_threshold=0.3,
            text_threshold=0.3,
            api_name="/predict"
    )

    return result

if __name__ == "__main__":
    alias = "bottle pack"
    print(request_vis(alias))