#!/bin/bash
set -euo pipefail

tmp_file="$(mktemp postman.json.XXXXXX)"

jq '
  def patch_login_request:
    if (type == "object")
      and (.request? and .request.url? and (.request.url.path? | join("/") == "auth/login"))
    then
      .request.body = {
        "mode": "urlencoded",
        "urlencoded": [
          {"key":"username","value":"{{USERNAME}}","type":"text"},
          {"key":"password","value":"{{PASSWORD}}","type":"text"}
        ]
      }
    else
      .
    end;
  def patch_forward_request:
    if (type == "object")
      and (.request? and .request.url? and (.request.url.path? | join("/") == "forward"))
    then
      .request.body = {
        "mode": "formdata",
        "formdata": [
          {"key":"image","type":"file","src":""}
        ]
      }
      | .request.header = (
          (.request.header // [])
          | map(
              if .key == "Content-Type"
              then . + {"disabled": true}
              else .
              end
            )
        )
    else
      .
    end;
  def patch_forward_multiple_request:
    if (type == "object")
      and (.request? and .request.url? and (.request.url.path? | join("/") == "forwardMultiple"))
    then
      .request.body = {
        "mode": "formdata",
        "formdata": [
          {"key":"images","type":"file","src":""}
        ]
      }
      | .request.header = (
          (.request.header // [])
          | map(
              if .key == "Content-Type"
              then . + {"disabled": true}
              elif .key == "X-Study-Ids"
              then . + {"disabled": false, "value": "{{STUDY_IDS}}"}
              else .
              end
            )
        )
    else
      .
    end;
  def walk(f):
    . as $in
    | if type == "object" then
        reduce keys[] as $key
          ({}; . + { ($key): ($in[$key] | walk(f)) })
        | f
      elif type == "array" then
        map(walk(f)) | f
      else
        f
      end;
  def replace_base_url:
    if type == "string"
    then gsub("\\{\\{baseUrl\\}\\}"; "{{BASE_URL}}")
    else .
    end;
  .auth = {
    "type": "bearer",
    "bearer": [
      {"key":"token","value":"{{TOKEN}}","type":"string"}
    ]
  }
  |
  .event = (
    (.event // []) + [
      {
        "listen":"prerequest",
        "script":{
          "type":"text/javascript",
          "exec":[
            "const token = pm.environment.get(\"TOKEN\");",
            "if (token) { return; }",
            "const baseUrl = pm.environment.get(\"BASE_URL\");",
            "const username = pm.environment.get(\"USERNAME\");",
            "const password = pm.environment.get(\"PASSWORD\");",
            "pm.sendRequest({",
            "  url: `${baseUrl}/auth/login`,",
            "  method: \"POST\",",
            "  header: { \"Content-Type\": \"application/x-www-form-urlencoded\" },",
            "  body: {",
            "    mode: \"urlencoded\",",
            "    urlencoded: [",
            "      { key: \"username\", value: username },",
            "      { key: \"password\", value: password }",
            "    ]",
            "  }",
            "}, (err, res) => {",
            "  if (err) { console.error(err); return; }",
            "  let json;",
            "  try { json = res.json(); } catch (e) {",
            "    console.error(\"Login response is not JSON\", res.text());",
            "    return;",
            "  }",
            "  if (!json || !json.access_token) {",
            "    console.error(\"Login response missing access_token\", json);",
            "    return;",
            "  }",
            "  pm.environment.set(\"TOKEN\", json.access_token);",
            "});"
          ]
        }
      }
    ]
  )
  | walk(patch_login_request)
  | walk(patch_forward_request)
  | walk(patch_forward_multiple_request)
  | walk(replace_base_url)
' postman.json > "${tmp_file}"

mv "${tmp_file}" postman.json
