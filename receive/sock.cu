#include"sock.h"
#include<stdexcept>
#include<netdb.h>
#include<string.h>
#include<sys/types.h>
#include <unistd.h> 

Sock::Sock()
{
     sock= socket(AF_INET, SOCK_STREAM, 0);
      if(sock == -1)
      {
           throw std::runtime_error("socketinit error\n");
      }
}
Sock::~Sock()
 {
      close(sock);
 }
bool Sock::Connect(const char *host_id, const int &port)
{
      server.sin_family = AF_INET;
      hp = gethostbyname(host_id);
      if(hp == (struct hostent *) 0)
      {
           std::cerr << "服务器地址获取失败！" << std::endl;
           return false;
      }
      memcpy((char *)&server.sin_addr,
      (char *)hp->h_addr, hp->h_length);
      server.sin_port = htons(port);
      if(connect(sock, (sockaddr *) &server,sizeof server) == -1)
      {
           std::cerr << "连接服务器失败！" << std::endl;
           return false;
      }
      else
            return true;
 }
 
 void Sock::send_socket(const char *s)
 {
      send(sock, s, strlen(s), 0);
 }
 int Sock::recv_socket()
 {
     memset(recvbuf,0,BUFSIZ);
     return recv(sock, recvbuf, BUFSIZ, 0);
  }

 std::string Sock::recv_socket1()
 {
	std::string buffer;
    char temp_buf[BUFSIZ];
    while (true) {
        int n = recv(sock, temp_buf, BUFSIZ, 0);
        if (n <= 0) {
            break;
        }
        buffer.append(temp_buf, n); // 将接收到的数据添加到字符串中
        if (n < BUFSIZ) {
            break;
        }
    }
    return buffer;
  }
  
 const char * Sock::get_recvbuf()
 {
      return recvbuf;
 }
