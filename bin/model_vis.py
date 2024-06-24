from gradio_client import Client


def request_vis(alias):
    client = Client("https://a508b0a8b407229dda.gradio.live/")
    result = client.predict(
            alias=alias,
            box_threshold=0.5,
            text_threshold=0.5,
            api_name="/predict"
    )

    return result

if __name__ == "__main__":
    alias = "bottle pack"
    print(request_vis(alias))