#include <ui.h>
#include <ui_web.h> //TODO: rename to ui_data or something
#include <mongoose.h>
#include <parson.h>
#include <raytracer.h>

static ui_ctx uctx;

//Mostly based off of the exampel code for the library.


static const char *s_http_port = "8000";
static struct mg_serve_http_opts s_http_server_opts;


void handle_ws_request(struct mg_connection *c, char* data)
{


    JSON_Value *root_value;
    JSON_Object *root_object;
	root_value = json_parse_string(data);
    root_object = json_value_get_object(root_value);

    switch((unsigned int)json_object_dotget_number(root_object, "type"))
    {
    case 0: //init
    {
        char buf[] = "{ \"type\":0, \"message\":\"Nothing Right Now.\"}";
        mg_send_websocket_frame(c, WEBSOCKET_OP_TEXT, buf, strlen(buf));

        return;
    }
    case 1: //action
    {
        switch((unsigned int)json_object_dotget_number(root_object, "action.type"))
        {
        case SS_RAYTRACER:
        {
            if(uctx.rctx->event_position==32)
                return;
            printf("UI Event Queued: Switch To Single Bounce\n");
            uctx.rctx->event_stack[uctx.rctx->event_position++] = SS_RAYTRACER;
            return;
        }
        case PATH_RAYTRACER: //prepass
        {
            if(uctx.rctx->event_position==32)
                return;
            printf("UI Event Queued: Switch To Path Raytracer\n");
            uctx.rctx->event_stack[uctx.rctx->event_position++] = PATH_RAYTRACER;
            return;
        }
        case SPLIT_PATH_RAYTRACER: //start render
        {
            if(uctx.rctx->event_position==32)
                return;
            printf("UI Event Queued: Switch To Split Path Raytracer\n");
            uctx.rctx->event_stack[uctx.rctx->event_position++] = SPLIT_PATH_RAYTRACER;
            return;
        }
        case 3: //start render
        {
            if(uctx.rctx->event_position==32)
                return;
            printf("Change Scene %s\n", json_object_dotget_string(root_object, "action.scene"));
            uctx.rctx->event_stack[uctx.rctx->event_position++] = 3;
            printf("Not supported\n");
            return;
        }
        }
        break;

    }
    case 2: //send kd tree to GE2
    {

        printf("GE2 requested k-d tree.\n");
        //char buf[] = "{ \"type\":0, \"message\":\"Nothing Right Now.\"}";
        if(uctx.rctx->stat_scene->kdt->buffer!=NULL)
        {

            mg_send_websocket_frame(c, WEBSOCKET_OP_TEXT, //TODO: put something for this (IT'S NOT TEXT)
                                    uctx.rctx->stat_scene->kdt->buffer,
                                    uctx.rctx->stat_scene->kdt->buffer_size);
        }
        else
            printf("ERROR: no k-d tree.\n");

        break;
    }
    }

}

static void ev_handler(struct mg_connection *c, int ev, void *p) {
    if (ev == MG_EV_HTTP_REQUEST) {
        struct http_message *hm = (struct http_message *) p;

        // We have received an HTTP request. Parsed request is contained in `hm`.
        // Send HTTP reply to the client which shows full original request.
        mg_send_head(c, 200, ___src_ui_index_html_len, "Content-Type: text/html");
        mg_printf(c, "%.*s", (int)___src_ui_index_html_len, ___src_ui_index_html);
    }
}


static void handle_ws(struct mg_connection *c, int ev, void* ev_data) {
    switch (ev)
    { //ignore confusing indentation
    case MG_EV_HTTP_REQUEST:
    {
        struct http_message *hm = (struct http_message *) ev_data;
        //TODO: do something here
        mg_send_head(c, 200, ___src_ui_index_html_len, "Content-Type: text/html");
        mg_printf(c, "%.*s", (int)___src_ui_index_html_len, ___src_ui_index_html);
        break;
    }
    case MG_EV_WEBSOCKET_HANDSHAKE_DONE:
    {
        printf("Webscoket Handshake\n");
        break;
    }
    case MG_EV_WEBSOCKET_FRAME:
    {
        struct websocket_message *wm = (struct websocket_message *) ev_data;
        /* New websocket message. Tell everybody. */
        //struct mg_str d = {(char *) wm->data, wm->size};
        //printf("WOW K: %s\n", wm->data);
        handle_ws_request(c, wm->data);
        break;
    }
    }

    //printf("TEST 3\n");
    //c->flags |= MG_F_SEND_AND_CLOSE;
}

static void handle_ocp_li(struct mg_connection *c, int ev, void* ev_data) {
    if (ev == MG_EV_HTTP_REQUEST) {
        struct http_message *hm = (struct http_message *) ev_data;

        // We have received an HTTP request. Parsed request is contained in `hm`.
        // Send HTTP reply to the client which shows full original request.
        mg_send_head(c, 200, ___src_ui_ocp_li_woff_len, "Content-Type: application/font-woff");
        //c->send_mbuf = ___src_ui_ocp_li_woff;
        //c->content_len = ___src_ui_ocp_li_woff_len;

        mg_send(c, ___src_ui_ocp_li_woff, ___src_ui_ocp_li_woff_len);
        //mg_printf(c, "%.*s", (int)___src_ui_ocp_li_woff_len, ___src_ui_ocp_li_woff);
    }
    //printf("TEST 2\n");
    c->flags |= MG_F_SEND_AND_CLOSE;
}


static void handle_style(struct mg_connection* c, int ev, void* ev_data) {
    if (ev == MG_EV_HTTP_REQUEST) {
        struct http_message *hm = (struct http_message *) ev_data;

        // We have received an HTTP request. Parsed request is contained in `hm`.
        // Send HTTP reply to the client which shows full original request.
        mg_send_head(c, 200, ___src_ui_style_css_len, "Content-Type: text/css");
        mg_printf(c, "%.*s", (int)___src_ui_style_css_len, ___src_ui_style_css);
    }
    //printf("TEST\n");
    c->flags |= MG_F_SEND_AND_CLOSE;
}

void web_server_start(void* rctx)
{
    uctx.rctx = rctx;
    struct mg_mgr mgr;
    struct mg_connection *c;

    mg_mgr_init(&mgr, NULL);
    c = mg_bind(&mgr, s_http_port, ev_handler);
    mg_set_protocol_http_websocket(c);
    mg_register_http_endpoint(c, "/ocp_li.woff", handle_ocp_li);
    mg_register_http_endpoint(c, "/style.css", handle_style);
    mg_register_http_endpoint(c, "/ws", handle_ws);

    printf("Web UI Hosted On Port %s\n", s_http_port);

    for (;;) {
        mg_mgr_poll(&mgr, 1000);
    }
    mg_mgr_free(&mgr);

    exit(1);

}
