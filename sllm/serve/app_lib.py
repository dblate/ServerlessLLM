# ---------------------------------------------------------------------------- #
#  serverlessllm                                                               #
#  copyright (c) serverlessllm team 2024                                       #
#                                                                              #
#  licensed under the apache license, version 2.0 (the "license");             #
#  you may not use this file except in compliance with the license.            #
#                                                                              #
#  you may obtain a copy of the license at                                     #
#                                                                              #
#                  http://www.apache.org/licenses/license-2.0                  #
#                                                                              #
#  unless required by applicable law or agreed to in writing, software         #
#  distributed under the license is distributed on an "as is" basis,           #
#  without warranties or conditions of any kind, either express or implied.    #
#  see the license for the specific language governing permissions and         #
#  limitations under the license.                                              #
# ---------------------------------------------------------------------------- #
import asyncio
import time
from contextlib import asynccontextmanager

import ray
import ray.exceptions
import shortuuid
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

from sllm.serve.logger import init_logger
from sllm.serve.openai_api_protocol import (
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    DeltaMessage,
)

logger = init_logger(__name__)


def chat_completion_stream_generator(model_name, generator):
    id = f"chatcmpl-{shortuuid.random()}"

    # first chunk with role
    choice_data = ChatCompletionResponseStreamChoice(
        index=0, delta=DeltaMessage(role="assistant"), finish_reason=None
    )
    chunk = ChatCompletionStreamResponse(
        id=id,
        choices=[choice_data],
        model=model_name,
    )
    yield f"data: {chunk.model_dump_json(exclude_unset=True)}\n\n"

    for val in generator:
        text = ray.get(val)
        if text == "":
            continue

        choice_data = ChatCompletionResponseStreamChoice(
            index=0,
            delta=DeltaMessage(content=text),
            finish_reason=None,
        )
        chunk = ChatCompletionStreamResponse(
            id=id, choices=[choice_data], model=model_name
        )
        yield f"data: {chunk.model_dump_json(exclude_unset=True)}\n\n"

    # last chunk for finish_reason
    choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(content=""),
        finish_reason="stop",
    )
    chunk = ChatCompletionStreamResponse(
        id=id, choices=[choice_data], model=model_name
    )
    yield f"data: {chunk.model_dump_json(exclude_unset=True)}\n\n"

    yield "data: [DONE]\n\n"


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Connect to the Ray cluster
        # ray.init()
        yield
        # Shutdown the Ray cluster
        ray.shutdown()

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    async def health_check():
        return {"status": "ok"}

    @app.post("/register")
    async def register_handler(request: Request):
        body = await request.json()

        controller = ray.get_actor("controller")
        if not controller:
            raise HTTPException(
                status_code=500, detail="Controller not initialized"
            )
        try:
            await controller.register.remote(body)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail="Cannot register model, please contact the administrator",
            )

        return {"status": "ok"}

    @app.post("/update")
    async def update_handler(request: Request):
        body = await request.json()
        model_name = body.get("model")
        if not model_name:
            raise HTTPException(
                status_code=400, detail="Missing model_name in request body"
            )

        controller = ray.get_actor("controller")
        if not controller:
            raise HTTPException(
                status_code=500, detail="Controller not initialized"
            )

        logger.info(f"Received request to update model {model_name}")
        try:
            await controller.update.remote(model_name, body)
        except ray.exceptions.RayTaskError as e:
            raise HTTPException(status_code=400, detail=str(e.cause))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

        return {"status": f"updated model {model_name}"}

    @app.post("/delete")
    async def delete_model(request: Request):
        body = await request.json()

        model_name = body.get("model")
        if not model_name:
            raise HTTPException(
                status_code=400, detail="Missing model_name in request body"
            )

        controller = ray.get_actor("controller")
        if not controller:
            raise HTTPException(
                status_code=500, detail="Controller not initialized"
            )

        logger.info(f"Received request to delete model {model_name}")
        await controller.delete.remote(model_name)

        return {"status": f"deleted model {model_name}"}

    @app.post("/v1/chat/completions_stream")
    async def generate_stream(request: Request):
        body = await request.json()
        model_name = body.get("model")
        logger.info(f"Received request for model {model_name}")
        if not model_name:
            raise HTTPException(
                status_code=400, detail="Missing model_name in request body"
            )

        request_router = ray.get_actor(model_name, namespace="models")
        logger.info(f"Got request router for {model_name}")

        generator = request_router.inference_stream.remote(body)
        final_generator = chat_completion_stream_generator(
            model_name, generator
        )
        return StreamingResponse(
            final_generator, media_type="text/event-stream"
        )

    async def inference_handler(request: Request, action: str):
        body = await request.json()
        model_name = body.get("model")
        logger.info(f"Received request for model {model_name}")
        if not model_name:
            raise HTTPException(
                status_code=400, detail="Missing model_name in request body"
            )

        request_router = ray.get_actor(model_name, namespace="models")
        logger.info(f"Got request router for {model_name}")

        result = request_router.inference.remote(body, action)
        return await result

    @app.post("/v1/chat/completions")
    async def generate_handler(request: Request):
        body = await request.json()
        model_name = body.get("model")
        logger.info(f"Received request for model {model_name}")
        if not model_name:
            raise HTTPException(
                status_code=400, detail="Missing model_name in request body"
            )

        request_router = ray.get_actor(model_name, namespace="models")
        logger.info(f"Got request router for {model_name}")

        if body.get("stream", False):
            generator = request_router.inference_stream.remote(body)
            final_generator = chat_completion_stream_generator(
                model_name, generator
            )
            return StreamingResponse(
                final_generator, media_type="text/event-stream"
            )

        result = request_router.inference.remote(body, "generate")
        return await result

    @app.post("/v1/embeddings")
    async def embeddings_handler(request: Request):
        return await inference_handler(request, "encode")

    @app.get("/v1/status")
    def status_handler(model):
        model_name = model
        logger.info(f"Received request for model {model_name}")
        if not model_name:
            raise HTTPException(
                status_code=400, detail="Missing model_name in request body"
            )

        request_router = ray.get_actor(model_name, namespace="models")
        logger.info(f"Got request router for {model_name}")
        status = request_router.get_status.remote()
        status = ray.get(status)
        return status

    return app
