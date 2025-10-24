from fastapi import FastAPI,Form,HTTPException,status
from typing import Tuple
from starlette.responses import JSONResponse
ALLOWED_OPERATIONS = {"sum","subtract","multiply","divide"}
app = FastAPI()
#数据预处理错误
def data_processing(operation:str,a:str,b:str)->Tuple[str,float,float]:
    if operation is None or a is None or b is None:
        raise  HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                             detail="数据输入残缺")
    operation=operation.strip().lower()
    if operation not in ALLOWED_OPERATIONS:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail="不支持格式的输入")
    try:
        data_a = float(a.strip())
        data_b = float(b.strip())
    except Exception:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail="data_a和data_B必须是数值")
    return operation,data_a,data_b

@app.post('/api')
#解析
async def calculation(
    operation:  str=  Form(...),
    Data_A:str=Form(...),
    Data_B:str=Form(...),
):
    try:
        operation,data_a,data_b=data_processing(operation,Data_A,Data_B)
#解析错误
    except HTTPException as e:
        raise e
    except Exception:
        raise HTTPException(status_code=500,detail="服务器在解析请求时发生错误")
#本地代码
    try:
        if operation == "sum":
            result = data_a + data_b
        elif operation == "subtract":
            result = data_a - data_b
        elif operation == "multiply":
            result = data_a * data_b
        elif operation == "divide":
            if data_b==0:
                raise HTTPException(status_code=400,detail="除法不能做分母")
            result = data_a / data_b
    except HTTPException as he:
        raise he
    except RuntimeError as re:
        raise HTTPException(status_code=500,detail=str(re))
    except Exception :
        raise HTTPException(status_code=500,detail="服务器在计算中发生错误")
#后处理
    if isinstance(result,float) and result.is_integer():
        result = int(result)
    else:
        result = result
#响应
    return JSONResponse(status_code=200,content={"Result":result})





